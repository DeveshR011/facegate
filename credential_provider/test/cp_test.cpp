// Drives FaceGateCP the way LogonUI does, against an in-process fake of the
// FaceGate service. Exits 0 when every check passes.
//
// Covers: class factory, scenario filtering, the automatic scan, auto-logon
// signalling, the packed KERB_INTERACTIVE_UNLOCK_LOGON blob, the failure and
// service-down paths, single-use credentials and UnAdvise during a scan.

#define WIN32_LEAN_AND_MEAN
#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS
#include <ntstatus.h>
#include <credentialprovider.h>
#include <ntsecapi.h>
#include <cstdio>
#include <string>

static const CLSID CLSID_FaceGateProvider =
    {0x7b3c9f2e, 0x5a41, 0x4d8b, {0x9e, 0x6c, 0x2f, 0x1a, 0x8d, 0x4b, 0x6c, 0x73}};
static const wchar_t kPipe[] = L"\\\\.\\pipe\\FaceGateUnlockTest";
static const wchar_t kSid[] = L"S-1-5-21-1111111111-2222222222-3333333333-1001";

static int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::printf("  FAIL line %d: %s\n", __LINE__, #cond);                \
        }                                                                        \
    } while (0)

// ---- fake LogonUI event sink ------------------------------------------------

class Events : public ICredentialProviderEvents {
public:
    HANDLE changed = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    LONG calls = 0;
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (riid == IID_IUnknown || riid == IID_ICredentialProviderEvents) {
            *ppv = this;
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    IFACEMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&_refs); }
    IFACEMETHODIMP_(ULONG) Release() override { return InterlockedDecrement(&_refs); }
    IFACEMETHODIMP CredentialsChanged(UINT_PTR context) override {
        CHECK(context == 42);
        InterlockedIncrement(&calls);
        SetEvent(changed);
        return S_OK;
    }

private:
    LONG _refs = 1;
};

// ---- fake service -----------------------------------------------------------

struct Server {
    std::wstring reply;
    DWORD delayMs = 0;
    HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE thread = nullptr;
    std::wstring request;
};

static DWORD WINAPI ServerThread(LPVOID param) {
    auto* s = static_cast<Server*>(param);
    HANDLE pipe = CreateNamedPipeW(kPipe, PIPE_ACCESS_DUPLEX,
                                   PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT, 1, 4096,
                                   4096, 0, nullptr);
    SetEvent(s->ready);
    if (pipe == INVALID_HANDLE_VALUE) return 1;
    if (ConnectNamedPipe(pipe, nullptr) || GetLastError() == ERROR_PIPE_CONNECTED) {
        wchar_t buffer[64] = {};
        DWORD n = 0;
        if (ReadFile(pipe, buffer, sizeof(buffer) - 2, &n, nullptr)) {
            s->request.assign(buffer, n / sizeof(wchar_t));
            Sleep(s->delayMs);
            WriteFile(pipe, s->reply.c_str(), static_cast<DWORD>(s->reply.size() * sizeof(wchar_t)),
                      &n, nullptr);
            FlushFileBuffers(pipe);
        }
    }
    DisconnectNamedPipe(pipe);
    CloseHandle(pipe);
    return 0;
}

static void StartServer(Server& s) {
    s.thread = CreateThread(nullptr, 0, ServerThread, &s, 0, nullptr);
    WaitForSingleObject(s.ready, 5000);
}

static void JoinServer(Server& s, DWORD timeout = 5000) {
    if (WaitForSingleObject(s.thread, timeout) != WAIT_OBJECT_0) {
        // Unblock a server nobody connected to.
        HANDLE h = CreateFileW(kPipe, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        WaitForSingleObject(s.thread, 2000);
    }
    CloseHandle(s.thread);
}

// ---- helpers ----------------------------------------------------------------

typedef HRESULT(STDAPICALLTYPE* GetClassObjectFn)(REFCLSID, REFIID, void**);
static GetClassObjectFn g_getClassObject = nullptr;

static ICredentialProvider* NewProvider() {
    IClassFactory* factory = nullptr;
    if (FAILED(g_getClassObject(CLSID_FaceGateProvider, IID_PPV_ARGS(&factory)))) return nullptr;
    ICredentialProvider* provider = nullptr;
    factory->CreateInstance(nullptr, IID_PPV_ARGS(&provider));
    factory->Release();
    return provider;
}

static std::wstring Unpack(const BYTE* blob, const UNICODE_STRING& s) {
    const auto offset = reinterpret_cast<ULONG_PTR>(s.Buffer);
    return std::wstring(reinterpret_cast<const wchar_t*>(blob + offset), s.Length / sizeof(wchar_t));
}

static std::wstring StatusOf(ICredentialProviderCredential* cred) {
    LPWSTR text = nullptr;
    std::wstring out;
    if (SUCCEEDED(cred->GetStringValue(2, &text)) && text) {
        out = text;
        CoTaskMemFree(text);
    }
    return out;
}

// ---- scenarios --------------------------------------------------------------

static void TestMatch() {
    std::printf("match -> auto-logon with a correctly packed credential\n");
    Server server;
    server.reply = L"OK\tLAPTOP\talice\tp@ss\tw0rd!";      // a tab inside the password must survive
    StartServer(server);

    Events events;
    ICredentialProvider* provider = NewProvider();
    CHECK(provider != nullptr);
    if (!provider) return;
    CHECK(provider->SetUsageScenario(CPUS_UNLOCK_WORKSTATION, 0) == S_OK);

    DWORD fields = 0;
    CHECK(provider->GetFieldDescriptorCount(&fields) == S_OK && fields == 4);
    for (DWORD i = 0; i < fields; ++i) {
        CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR* d = nullptr;
        CHECK(provider->GetFieldDescriptorAt(i, &d) == S_OK && d && d->dwFieldID == i);
        if (d) {
            CoTaskMemFree(d->pszLabel);
            CoTaskMemFree(d);
        }
    }

    DWORD count = 0, def = 0;
    BOOL autoLogon = TRUE;
    CHECK(provider->GetCredentialCount(&count, &def, &autoLogon) == S_OK);
    CHECK(count == 1 && autoLogon == FALSE && def == CREDENTIAL_PROVIDER_NO_DEFAULT);

    CHECK(provider->Advise(&events, 42) == S_OK);
    CHECK(WaitForSingleObject(events.changed, 5000) == WAIT_OBJECT_0);
    JoinServer(server);
    CHECK(server.request == L"AUTH");

    CHECK(provider->GetCredentialCount(&count, &def, &autoLogon) == S_OK);
    CHECK(count == 1 && def == 0 && autoLogon == TRUE);

    ICredentialProviderCredential* cred = nullptr;
    CHECK(provider->GetCredentialAt(0, &cred) == S_OK && cred);
    if (cred) {
        ICredentialProviderCredential2* cred2 = nullptr;
        CHECK(cred->QueryInterface(IID_PPV_ARGS(&cred2)) == S_OK);
        if (cred2) {
            LPWSTR sid = nullptr;
            CHECK(cred2->GetUserSid(&sid) == S_OK && sid && wcscmp(sid, kSid) == 0);
            CoTaskMemFree(sid);
            cred2->Release();
        }
        HBITMAP bitmap = nullptr;
        CHECK(cred->GetBitmapValue(0, &bitmap) == S_OK && bitmap);
        if (bitmap) DeleteObject(bitmap);

        BOOL selectedAuto = FALSE;
        CHECK(cred->SetSelected(&selectedAuto) == S_OK && selectedAuto == TRUE);

        CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE response;
        CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION ser;
        LPWSTR text = nullptr;
        CREDENTIAL_PROVIDER_STATUS_ICON icon;
        CHECK(cred->GetSerialization(&response, &ser, &text, &icon) == S_OK);
        CHECK(response == CPGSR_RETURN_CREDENTIAL_FINISHED);
        CHECK(ser.clsidCredentialProvider == CLSID_FaceGateProvider);
        CHECK(ser.cbSerialization > sizeof(KERB_INTERACTIVE_UNLOCK_LOGON) && ser.rgbSerialization);
        if (response == CPGSR_RETURN_CREDENTIAL_FINISHED && ser.rgbSerialization) {
            auto* k = reinterpret_cast<const KERB_INTERACTIVE_UNLOCK_LOGON*>(ser.rgbSerialization);
            CHECK(k->Logon.MessageType == KerbWorkstationUnlockLogon);
            CHECK(Unpack(ser.rgbSerialization, k->Logon.LogonDomainName) == L"LAPTOP");
            CHECK(Unpack(ser.rgbSerialization, k->Logon.UserName) == L"alice");
            CHECK(Unpack(ser.rgbSerialization, k->Logon.Password) == L"p@ss\tw0rd!");
            const size_t end = reinterpret_cast<ULONG_PTR>(k->Logon.Password.Buffer) + k->Logon.Password.Length;
            CHECK(end == ser.cbSerialization);
            CoTaskMemFree(ser.rgbSerialization);
        }

        // The credential is single use: a second request must come back empty.
        CHECK(cred->GetSerialization(&response, &ser, &text, &icon) == S_OK);
        CHECK(response == CPGSR_NO_CREDENTIAL_NOT_FINISHED && ser.rgbSerialization == nullptr);
        CHECK(provider->GetCredentialCount(&count, &def, &autoLogon) == S_OK && autoLogon == FALSE);
        cred->Release();
    }
    CHECK(provider->UnAdvise() == S_OK);
    provider->Release();
}

static void TestLogonScenarioMessageType() {
    std::printf("CPUS_LOGON packs KerbInteractiveLogon\n");
    Server server;
    server.reply = L"OK\tMicrosoftAccount\tbob@example.com\tsecret";
    StartServer(server);
    Events events;
    ICredentialProvider* provider = NewProvider();
    CHECK(provider->SetUsageScenario(CPUS_LOGON, 0) == S_OK);
    provider->Advise(&events, 42);
    CHECK(WaitForSingleObject(events.changed, 5000) == WAIT_OBJECT_0);
    JoinServer(server);
    ICredentialProviderCredential* cred = nullptr;
    CHECK(provider->GetCredentialAt(0, &cred) == S_OK);
    if (cred) {
        CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE response;
        CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION ser;
        CHECK(cred->GetSerialization(&response, &ser, nullptr, nullptr) == S_OK);
        CHECK(response == CPGSR_RETURN_CREDENTIAL_FINISHED);
        if (ser.rgbSerialization) {
            auto* k = reinterpret_cast<const KERB_INTERACTIVE_UNLOCK_LOGON*>(ser.rgbSerialization);
            CHECK(k->Logon.MessageType == KerbInteractiveLogon);
            CHECK(Unpack(ser.rgbSerialization, k->Logon.UserName) == L"bob@example.com");
            CoTaskMemFree(ser.rgbSerialization);
        }
        cred->Release();
    }
    provider->UnAdvise();
    provider->Release();
}

static void TestNoMatch() {
    std::printf("no match -> tile stays, reason shown, no credential\n");
    Server server;
    server.reply = L"FAIL\tFace not recognised";
    StartServer(server);
    Events events;
    ICredentialProvider* provider = NewProvider();
    provider->SetUsageScenario(CPUS_UNLOCK_WORKSTATION, 0);
    provider->Advise(&events, 42);
    CHECK(WaitForSingleObject(events.changed, 5000) == WAIT_OBJECT_0);
    JoinServer(server);

    DWORD count = 0, def = 0;
    BOOL autoLogon = TRUE;
    CHECK(provider->GetCredentialCount(&count, &def, &autoLogon) == S_OK);
    CHECK(count == 1 && autoLogon == FALSE);
    ICredentialProviderCredential* cred = nullptr;
    CHECK(provider->GetCredentialAt(0, &cred) == S_OK);
    if (cred) {
        CHECK(StatusOf(cred) == L"Face not recognised");
        CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE response;
        CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION ser;
        CHECK(cred->GetSerialization(&response, &ser, nullptr, nullptr) == S_OK);
        CHECK(response == CPGSR_NO_CREDENTIAL_NOT_FINISHED);

        // "Try again" starts exactly one more scan.
        Server retry;
        retry.reply = L"FAIL\tToo many attempts - use your password";
        StartServer(retry);
        CHECK(cred->CommandLinkClicked(3) == S_OK);
        CHECK(WaitForSingleObject(events.changed, 5000) == WAIT_OBJECT_0);
        JoinServer(retry);
        CHECK(StatusOf(cred) == L"Too many attempts - use your password");
        CHECK(events.calls == 2);
        cred->Release();
    }
    provider->UnAdvise();
    provider->Release();
}

static void TestServiceDown() {
    std::printf("service not running -> quiet failure\n");
    Events events;
    ICredentialProvider* provider = NewProvider();
    provider->SetUsageScenario(CPUS_UNLOCK_WORKSTATION, 0);
    provider->Advise(&events, 42);
    CHECK(WaitForSingleObject(events.changed, 5000) == WAIT_OBJECT_0);
    DWORD count = 0, def = 0;
    BOOL autoLogon = TRUE;
    provider->GetCredentialCount(&count, &def, &autoLogon);
    CHECK(autoLogon == FALSE);
    ICredentialProviderCredential* cred = nullptr;
    if (SUCCEEDED(provider->GetCredentialAt(0, &cred)) && cred) {
        CHECK(StatusOf(cred) == L"FaceGate is not running");
        cred->Release();
    }
    provider->UnAdvise();
    provider->Release();
}

static void TestMalformedReply() {
    std::printf("malformed OK reply -> rejected\n");
    Server server;
    server.reply = L"OK\tLAPTOP";                 // no user, no password
    StartServer(server);
    Events events;
    ICredentialProvider* provider = NewProvider();
    provider->SetUsageScenario(CPUS_UNLOCK_WORKSTATION, 0);
    provider->Advise(&events, 42);
    CHECK(WaitForSingleObject(events.changed, 5000) == WAIT_OBJECT_0);
    JoinServer(server);
    DWORD count = 0, def = 0;
    BOOL autoLogon = TRUE;
    provider->GetCredentialCount(&count, &def, &autoLogon);
    CHECK(autoLogon == FALSE);
    provider->UnAdvise();
    provider->Release();
}

static void TestUnadviseDuringScan() {
    std::printf("UnAdvise while scanning -> no callback, result dropped\n");
    Server server;
    server.reply = L"OK\tLAPTOP\talice\tpw";
    server.delayMs = 600;
    StartServer(server);
    Events events;
    ICredentialProvider* provider = NewProvider();
    provider->SetUsageScenario(CPUS_UNLOCK_WORKSTATION, 0);
    provider->Advise(&events, 42);
    Sleep(150);                                    // the worker is now blocked in ReadFile
    CHECK(provider->UnAdvise() == S_OK);
    provider->Release();                           // the worker still holds a reference
    CHECK(WaitForSingleObject(events.changed, 1200) == WAIT_TIMEOUT);
    CHECK(events.calls == 0);
    JoinServer(server);
}

static void TestOtherScenarios() {
    std::printf("UAC / change-password scenarios -> declined\n");
    ICredentialProvider* provider = NewProvider();
    CHECK(provider->SetUsageScenario(CPUS_CREDUI, 0) == E_NOTIMPL);
    CHECK(provider->SetUsageScenario(CPUS_CHANGE_PASSWORD, 0) == E_NOTIMPL);
    provider->Release();

    IClassFactory* factory = nullptr;
    const CLSID other = {0x11111111, 0x2222, 0x3333, {0, 1, 2, 3, 4, 5, 6, 7}};
    CHECK(g_getClassObject(other, IID_PPV_ARGS(&factory)) == CLASS_E_CLASSNOTAVAILABLE);
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) {
        std::printf("usage: cp_test <path to FaceGateCP_test.dll>\n");
        return 2;
    }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    HMODULE dll = LoadLibraryW(argv[1]);
    if (!dll) {
        std::printf("could not load %ls (error %lu)\n", argv[1], GetLastError());
        return 2;
    }
    g_getClassObject = reinterpret_cast<GetClassObjectFn>(GetProcAddress(dll, "DllGetClassObject"));
    auto canUnload = reinterpret_cast<HRESULT(STDAPICALLTYPE*)()>(GetProcAddress(dll, "DllCanUnloadNow"));
    if (!g_getClassObject || !canUnload) {
        std::printf("exports missing\n");
        return 2;
    }

    TestOtherScenarios();
    TestMatch();
    TestLogonScenarioMessageType();
    TestNoMatch();
    TestServiceDown();
    TestMalformedReply();
    TestUnadviseDuringScan();

    Sleep(300);                                    // let the last worker thread exit
    std::printf("all objects released -> DLL can unload\n");
    CHECK(canUnload() == S_OK);

    std::printf(g_failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", g_failures);
    return g_failures ? 1 : 0;
}
