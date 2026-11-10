// FaceGate credential provider.
//
// Loaded by LogonUI on the lock screen. It owns no recognition logic: it asks
// the FaceGate service (running as SYSTEM) over a named pipe whether the owner
// is in front of the camera, and if so serialises the credential the service
// returns so Windows can unlock the session.
//
//   LogonUI ──loads──> FaceGateCP.dll ──\\.\pipe\FaceGateUnlock──> facegate service
//
// Design rules
//   * Never block LogonUI's UI thread: the pipe conversation runs on a worker.
//   * One automatic scan per appearance of the lock screen; further scans only
//     when the user clicks "Try again". A stale stored password therefore
//     cannot cause a logon-failure loop.
//   * If anything is missing or fails, expose no credential at all. The
//     built-in password / PIN providers are untouched.

#define WIN32_LEAN_AND_MEAN
#define SECURITY_WIN32
#define WIN32_NO_STATUS          // let ntstatus.h own the STATUS_* macros
#include <windows.h>
#undef WIN32_NO_STATUS
#include <ntstatus.h>
#include <unknwn.h>
#include <initguid.h>            // instantiate the CPFG_* field GUIDs in this module
#include <shlguid.h>
#include <credentialprovider.h>
#include <ntsecapi.h>
#include <security.h>
#include <intsafe.h>
#include <shlwapi.h>
#include <objbase.h>
#include <gdiplus.h>
#include <new>
#include <string>

#pragma comment(lib, "secur32.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "advapi32.lib")

#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS ((NTSTATUS)0x00000000L)
#endif

// {7B3C9F2E-5A41-4D8B-9E6C-2F1A8D4B6C73}
static const CLSID CLSID_FaceGateProvider =
    {0x7b3c9f2e, 0x5a41, 0x4d8b, {0x9e, 0x6c, 0x2f, 0x1a, 0x8d, 0x4b, 0x6c, 0x73}};

// The test build (test/cp_test.cpp) compiles this file with its own pipe name
// and a fixed SID, so the harness needs neither the real service nor HKLM.
// The shipped DLL contains no such switch.
#ifdef FACEGATE_TEST_BUILD
static const wchar_t kPipeName[] = L"\\\\.\\pipe\\FaceGateUnlockTest";
#else
static const wchar_t kPipeName[] = L"\\\\.\\pipe\\FaceGateUnlock";
#endif
static const wchar_t kRegKey[] = L"SOFTWARE\\FaceGate";

static HINSTANCE g_hinst = nullptr;
static LONG g_refs = 0;

static void DllAddRef() { InterlockedIncrement(&g_refs); }
static void DllRelease() { InterlockedDecrement(&g_refs); }

// ---------------------------------------------------------------------------
// Fields shown on the tile
// ---------------------------------------------------------------------------

enum FieldId { FI_IMAGE = 0, FI_LABEL, FI_STATUS, FI_RETRY, FI_COUNT };

static const CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR kFields[FI_COUNT] = {
    {FI_IMAGE, CPFT_TILE_IMAGE, const_cast<LPWSTR>(L"Image"), CPFG_CREDENTIAL_PROVIDER_LOGO},
    {FI_LABEL, CPFT_SMALL_TEXT, const_cast<LPWSTR>(L"Label"), CPFG_CREDENTIAL_PROVIDER_LABEL},
    {FI_STATUS, CPFT_LARGE_TEXT, const_cast<LPWSTR>(L"Status"), {0}},
    {FI_RETRY, CPFT_COMMAND_LINK, const_cast<LPWSTR>(L"Try again"), {0}},
};

static const CREDENTIAL_PROVIDER_FIELD_STATE kFieldStates[FI_COUNT] = {
    CPFS_DISPLAY_IN_BOTH, CPFS_HIDDEN, CPFS_DISPLAY_IN_SELECTED_TILE, CPFS_DISPLAY_IN_SELECTED_TILE};

// ---------------------------------------------------------------------------
// Serialisation helpers (after Microsoft's credential provider samples)
// ---------------------------------------------------------------------------

static HRESULT UnicodeStringInit(PWSTR text, UNICODE_STRING* out) {
    if (!text) return E_INVALIDARG;
    size_t bytes = wcslen(text) * sizeof(wchar_t);
    if (bytes > 0xFFFF) return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
    out->Length = static_cast<USHORT>(bytes);
    out->MaximumLength = static_cast<USHORT>(bytes);
    out->Buffer = text;
    return S_OK;
}

static void PackString(const UNICODE_STRING& in, BYTE* base, BYTE** cursor, UNICODE_STRING* out) {
    out->Length = in.Length;
    out->MaximumLength = in.Length;
    CopyMemory(*cursor, in.Buffer, in.Length);
    // In the packed form "Buffer" holds an offset from the start of the blob.
    out->Buffer = reinterpret_cast<PWSTR>(static_cast<ULONG_PTR>(*cursor - base));
    *cursor += in.Length;
}

// Builds the flat KERB_INTERACTIVE_UNLOCK_LOGON blob LSA expects. CoTaskMem-allocated.
static HRESULT PackUnlockLogon(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, PWSTR domain, PWSTR user,
                               PWSTR password, BYTE** blob, DWORD* blobSize) {
    KERB_INTERACTIVE_UNLOCK_LOGON in;
    ZeroMemory(&in, sizeof(in));
    HRESULT hr = UnicodeStringInit(domain, &in.Logon.LogonDomainName);
    if (SUCCEEDED(hr)) hr = UnicodeStringInit(user, &in.Logon.UserName);
    if (SUCCEEDED(hr)) hr = UnicodeStringInit(password, &in.Logon.Password);
    if (FAILED(hr)) return hr;
    in.Logon.MessageType =
        (cpus == CPUS_UNLOCK_WORKSTATION) ? KerbWorkstationUnlockLogon : KerbInteractiveLogon;

    DWORD size = sizeof(in) + in.Logon.LogonDomainName.Length + in.Logon.UserName.Length +
                 in.Logon.Password.Length;
    BYTE* out = static_cast<BYTE*>(CoTaskMemAlloc(size));
    if (!out) return E_OUTOFMEMORY;
    ZeroMemory(out, size);

    auto* packed = reinterpret_cast<KERB_INTERACTIVE_UNLOCK_LOGON*>(out);
    packed->Logon.MessageType = in.Logon.MessageType;
    BYTE* cursor = out + sizeof(*packed);
    PackString(in.Logon.LogonDomainName, out, &cursor, &packed->Logon.LogonDomainName);
    PackString(in.Logon.UserName, out, &cursor, &packed->Logon.UserName);
    PackString(in.Logon.Password, out, &cursor, &packed->Logon.Password);

    *blob = out;
    *blobSize = size;
    return S_OK;
}

static HRESULT NegotiateAuthPackage(ULONG* package) {
    HANDLE lsa = nullptr;
    NTSTATUS status = LsaConnectUntrusted(&lsa);
    if (status != STATUS_SUCCESS) return HRESULT_FROM_NT(status);
    char name[] = NEGOSSP_NAME_A;
    LSA_STRING lsaName;
    lsaName.Buffer = name;
    lsaName.Length = static_cast<USHORT>(strlen(name));
    lsaName.MaximumLength = lsaName.Length + 1;
    status = LsaLookupAuthenticationPackage(lsa, &lsaName, package);
    LsaDeregisterLogonProcess(lsa);
    return status == STATUS_SUCCESS ? S_OK : HRESULT_FROM_NT(status);
}

static void WipeString(std::wstring& s) {
    if (!s.empty()) SecureZeroMemory(&s[0], s.size() * sizeof(wchar_t));
    s.clear();
}

static std::wstring ReadTargetSid() {
#ifdef FACEGATE_TEST_BUILD
    return L"S-1-5-21-1111111111-2222222222-3333333333-1001";
#else
    wchar_t buffer[256];
    DWORD size = sizeof(buffer);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, kRegKey, L"UserSid", RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                     nullptr, buffer, &size) != ERROR_SUCCESS)
        return L"";
    return buffer;
#endif
}

// ---------------------------------------------------------------------------
// Tile image: the face glyph on an accent-blue rounded square
// ---------------------------------------------------------------------------

static HBITMAP CreateTileBitmap() {
    using namespace Gdiplus;
    GdiplusStartupInput startup;
    ULONG_PTR token = 0;
    if (GdiplusStartup(&token, &startup, nullptr) != Ok) return nullptr;
    HBITMAP result = nullptr;
    {
        const int size = 192;
        Bitmap bitmap(size, size, PixelFormat32bppARGB);
        Graphics g(&bitmap);
        g.SetSmoothingMode(SmoothingModeAntiAlias);
        g.Clear(Color(255, 10, 132, 255));

        Pen pen(Color(255, 255, 255, 255), size * 0.052f);
        pen.SetStartCap(LineCapRound);
        pen.SetEndCap(LineCapRound);
        pen.SetLineJoin(LineJoinRound);

        const float c = size / 2.0f, h = size * 0.34f, r = size * 0.115f, arm = size * 0.09f;
        for (int i = 0; i < 4; ++i) {
            const float sx = (i & 1) ? -1.0f : 1.0f, sy = (i & 2) ? -1.0f : 1.0f;
            GraphicsPath path;
            const float x0 = c - sx * h, y0 = c - sy * h;   // outer corner
            path.AddLine(x0, y0 + sy * (r + arm), x0, y0 + sy * r);
            path.AddBezier(x0, y0 + sy * r, x0, y0 + sy * r * 0.45f, x0 + sx * r * 0.45f, y0,
                           x0 + sx * r, y0);
            path.AddLine(x0 + sx * r, y0, x0 + sx * (r + arm), y0);
            g.DrawPath(&pen, &path);
        }
        const float u = size * 0.85f;                        // face features scale
        g.DrawLine(&pen, c - 0.150f * u, c - 0.135f * u, c - 0.150f * u, c - 0.050f * u);
        g.DrawLine(&pen, c + 0.150f * u, c - 0.135f * u, c + 0.150f * u, c - 0.050f * u);
        GraphicsPath nose;
        nose.AddLine(c + 0.012f * u, c - 0.135f * u, c + 0.012f * u, c + 0.050f * u);
        nose.AddBezier(c + 0.012f * u, c + 0.050f * u, c + 0.012f * u, c + 0.098f * u,
                       c - 0.020f * u, c + 0.098f * u, c - 0.052f * u, c + 0.098f * u);
        g.DrawPath(&pen, &nose);
        g.DrawBezier(&pen, c - 0.150f * u, c + 0.180f * u, c - 0.075f * u, c + 0.262f * u,
                     c + 0.075f * u, c + 0.262f * u, c + 0.150f * u, c + 0.180f * u);

        bitmap.GetHBITMAP(Color(255, 10, 132, 255), &result);
    }
    GdiplusShutdown(token);
    return result;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

class Credential;

class Provider : public ICredentialProvider, public ICredentialProviderSetUserArray {
public:
    Provider();

    // IUnknown
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override;
    IFACEMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&_refs); }
    IFACEMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&_refs);
        if (!n) delete this;
        return n;
    }

    // ICredentialProvider
    IFACEMETHODIMP SetUsageScenario(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, DWORD flags) override;
    IFACEMETHODIMP SetSerialization(const CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION*) override {
        return E_NOTIMPL;
    }
    IFACEMETHODIMP Advise(ICredentialProviderEvents* events, UINT_PTR context) override;
    IFACEMETHODIMP UnAdvise() override;
    IFACEMETHODIMP GetFieldDescriptorCount(DWORD* count) override;
    IFACEMETHODIMP GetFieldDescriptorAt(DWORD index,
                                        CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR** descriptor) override;
    IFACEMETHODIMP GetCredentialCount(DWORD* count, DWORD* defaultIndex,
                                      BOOL* autoLogon) override;
    IFACEMETHODIMP GetCredentialAt(DWORD index, ICredentialProviderCredential** credential) override;

    // ICredentialProviderSetUserArray
    IFACEMETHODIMP SetUserArray(ICredentialProviderUserArray* users) override;

    // Used by Credential
    CREDENTIAL_PROVIDER_USAGE_SCENARIO Scenario() const { return _cpus; }
    const std::wstring& TargetSid() const { return _sid; }
    std::wstring Status();
    bool Matched();
    bool TakeCredential(std::wstring& domain, std::wstring& user, std::wstring& password);
    void StartScan();
    void ReportRejected();

private:
    ~Provider();
    static DWORD WINAPI ScanThread(LPVOID param);
    void Scan();
    void Finish(bool matched, const std::wstring& status);

    LONG _refs = 1;
    CREDENTIAL_PROVIDER_USAGE_SCENARIO _cpus = CPUS_INVALID;
    bool _enabled = false;
    bool _userPresent = true;
    std::wstring _sid;
    Credential* _credential = nullptr;

    SRWLOCK _lock = SRWLOCK_INIT;                 // guards everything below
    ICredentialProviderEvents* _events = nullptr;
    UINT_PTR _adviseContext = 0;
    bool _scanning = false;
    bool _matched = false;
    bool _cancel = false;
    HANDLE _pipe = INVALID_HANDLE_VALUE;
    std::wstring _status = L"Looking for you\x2026";
    std::wstring _domain, _user, _password;
};

// ---------------------------------------------------------------------------
// Credential (the tile)
// ---------------------------------------------------------------------------

class Credential : public ICredentialProviderCredential2 {
public:
    explicit Credential(Provider* provider) : _provider(provider) { DllAddRef(); }

    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_ICredentialProviderCredential ||
            riid == IID_ICredentialProviderCredential2) {
            *ppv = static_cast<ICredentialProviderCredential2*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    IFACEMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&_refs); }
    IFACEMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&_refs);
        if (!n) delete this;
        return n;
    }

    void Detach() { _provider = nullptr; }

    IFACEMETHODIMP Advise(ICredentialProviderCredentialEvents* events) override {
        if (_events) _events->Release();
        _events = events;
        if (_events) _events->AddRef();
        return S_OK;
    }
    IFACEMETHODIMP UnAdvise() override {
        if (_events) _events->Release();
        _events = nullptr;
        return S_OK;
    }
    IFACEMETHODIMP SetSelected(BOOL* autoLogon) override {
        *autoLogon = (_provider && _provider->Matched()) ? TRUE : FALSE;
        return S_OK;
    }
    IFACEMETHODIMP SetDeselected() override { return S_OK; }

    IFACEMETHODIMP GetFieldState(DWORD field, CREDENTIAL_PROVIDER_FIELD_STATE* state,
                                 CREDENTIAL_PROVIDER_FIELD_INTERACTIVE_STATE* interactive) override {
        if (field >= FI_COUNT || !state || !interactive) return E_INVALIDARG;
        *state = kFieldStates[field];
        *interactive = CPFIS_NONE;
        return S_OK;
    }
    IFACEMETHODIMP GetStringValue(DWORD field, LPWSTR* value) override {
        if (field >= FI_COUNT || !value) return E_INVALIDARG;
        switch (field) {
            case FI_LABEL: return SHStrDupW(L"FaceGate", value);
            case FI_STATUS:
                return SHStrDupW(_provider ? _provider->Status().c_str() : L"", value);
            case FI_RETRY: return SHStrDupW(L"Try again", value);
            default: return SHStrDupW(L"", value);
        }
    }
    IFACEMETHODIMP GetBitmapValue(DWORD field, HBITMAP* bitmap) override {
        if (field != FI_IMAGE || !bitmap) return E_INVALIDARG;
        *bitmap = CreateTileBitmap();
        return *bitmap ? S_OK : E_FAIL;
    }
    IFACEMETHODIMP GetCheckboxValue(DWORD, BOOL*, LPWSTR*) override { return E_NOTIMPL; }
    IFACEMETHODIMP GetSubmitButtonValue(DWORD, DWORD*) override { return E_NOTIMPL; }
    IFACEMETHODIMP GetComboBoxValueCount(DWORD, DWORD*, DWORD*) override { return E_NOTIMPL; }
    IFACEMETHODIMP GetComboBoxValueAt(DWORD, DWORD, LPWSTR*) override { return E_NOTIMPL; }
    IFACEMETHODIMP SetStringValue(DWORD, LPCWSTR) override { return E_NOTIMPL; }
    IFACEMETHODIMP SetCheckboxValue(DWORD, BOOL) override { return E_NOTIMPL; }
    IFACEMETHODIMP SetComboBoxSelectedValue(DWORD, DWORD) override { return E_NOTIMPL; }
    IFACEMETHODIMP CommandLinkClicked(DWORD field) override {
        if (field != FI_RETRY) return E_INVALIDARG;
        if (_provider) _provider->StartScan();
        return S_OK;
    }

    IFACEMETHODIMP GetSerialization(CREDENTIAL_PROVIDER_GET_SERIALIZATION_RESPONSE* response,
                                    CREDENTIAL_PROVIDER_CREDENTIAL_SERIALIZATION* serialization,
                                    LPWSTR* statusText,
                                    CREDENTIAL_PROVIDER_STATUS_ICON* statusIcon) override {
        *response = CPGSR_NO_CREDENTIAL_NOT_FINISHED;
        if (statusText) *statusText = nullptr;
        if (statusIcon) *statusIcon = CPSI_NONE;
        ZeroMemory(serialization, sizeof(*serialization));

        std::wstring domain, user, password;
        if (!_provider || !_provider->TakeCredential(domain, user, password)) return S_OK;

        BYTE* blob = nullptr;
        DWORD blobSize = 0;
        ULONG package = 0;
        HRESULT hr = PackUnlockLogon(_provider->Scenario(), &domain[0], &user[0], &password[0],
                                     &blob, &blobSize);
        WipeString(password);
        if (SUCCEEDED(hr)) hr = NegotiateAuthPackage(&package);
        if (FAILED(hr)) {
            if (blob) {
                SecureZeroMemory(blob, blobSize);
                CoTaskMemFree(blob);
            }
            return S_OK;      // no credential; the user can still use another sign-in option
        }
        serialization->ulAuthenticationPackage = package;
        serialization->clsidCredentialProvider = CLSID_FaceGateProvider;
        serialization->rgbSerialization = blob;
        serialization->cbSerialization = blobSize;
        *response = CPGSR_RETURN_CREDENTIAL_FINISHED;
        return S_OK;
    }

    IFACEMETHODIMP ReportResult(NTSTATUS status, NTSTATUS, LPWSTR* statusText,
                                CREDENTIAL_PROVIDER_STATUS_ICON* statusIcon) override {
        if (statusText) *statusText = nullptr;
        if (statusIcon) *statusIcon = CPSI_NONE;
        if (status != STATUS_SUCCESS && _provider) {
            // Windows refused what the service handed us (password changed since
            // setup). Tell the service so it stops offering it.
            _provider->ReportRejected();
            if (statusText)
                SHStrDupW(L"Windows rejected the saved password. Sign in with your password, then "
                          L"turn Windows unlock off and on again in FaceGate.",
                          statusText);
            if (statusIcon) *statusIcon = CPSI_ERROR;
        }
        return S_OK;
    }

    // ICredentialProviderCredential2
    IFACEMETHODIMP GetUserSid(LPWSTR* sid) override {
        if (!sid) return E_POINTER;
        if (!_provider || _provider->TargetSid().empty()) {
            *sid = nullptr;
            return E_UNEXPECTED;
        }
        return SHStrDupW(_provider->TargetSid().c_str(), sid);
    }

private:
    ~Credential() {
        if (_events) _events->Release();
        DllRelease();
    }
    LONG _refs = 1;
    Provider* _provider;       // not owned; cleared by Provider::~Provider via Detach()
    ICredentialProviderCredentialEvents* _events = nullptr;
};

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

Provider::Provider() {
    DllAddRef();
    _sid = ReadTargetSid();
}

Provider::~Provider() {
    AcquireSRWLockExclusive(&_lock);
    WipeString(_password);
    ReleaseSRWLockExclusive(&_lock);
    if (_credential) {
        _credential->Detach();
        _credential->Release();
    }
    DllRelease();
}

IFACEMETHODIMP Provider::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_ICredentialProvider) {
        *ppv = static_cast<ICredentialProvider*>(this);
    } else if (riid == IID_ICredentialProviderSetUserArray) {
        *ppv = static_cast<ICredentialProviderSetUserArray*>(this);
    } else {
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

IFACEMETHODIMP Provider::SetUsageScenario(CREDENTIAL_PROVIDER_USAGE_SCENARIO cpus, DWORD) {
    _cpus = cpus;
    // Lock screen only (Windows 10+ reports an unlock as either scenario).
    // Not for UAC prompts (CPUS_CREDUI) or password changes.
    if (cpus != CPUS_LOGON && cpus != CPUS_UNLOCK_WORKSTATION) return E_NOTIMPL;
    _enabled = !_sid.empty();
    if (_enabled && !_credential) _credential = new (std::nothrow) Credential(this);
    if (!_credential) _enabled = false;
    return S_OK;
}

IFACEMETHODIMP Provider::SetUserArray(ICredentialProviderUserArray* users) {
    // Offer the tile only when the enrolled account is one of the users on screen.
    _userPresent = false;
    DWORD count = 0;
    if (users && SUCCEEDED(users->GetCount(&count))) {
        for (DWORD i = 0; i < count && !_userPresent; ++i) {
            ICredentialProviderUser* user = nullptr;
            if (SUCCEEDED(users->GetAt(i, &user)) && user) {
                LPWSTR sid = nullptr;
                if (SUCCEEDED(user->GetSid(&sid)) && sid) {
                    _userPresent = (_wcsicmp(sid, _sid.c_str()) == 0);
                    CoTaskMemFree(sid);
                }
                user->Release();
            }
        }
    }
    return S_OK;
}

IFACEMETHODIMP Provider::Advise(ICredentialProviderEvents* events, UINT_PTR context) {
    AcquireSRWLockExclusive(&_lock);
    if (_events) _events->Release();
    _events = events;
    if (_events) _events->AddRef();
    _adviseContext = context;
    ReleaseSRWLockExclusive(&_lock);
    if (_enabled && _userPresent) StartScan();      // the one automatic scan
    return S_OK;
}

IFACEMETHODIMP Provider::UnAdvise() {
    AcquireSRWLockExclusive(&_lock);
    if (_events) _events->Release();
    _events = nullptr;
    _cancel = true;
    if (_pipe != INVALID_HANDLE_VALUE) CancelIoEx(_pipe, nullptr);   // wake the worker
    ReleaseSRWLockExclusive(&_lock);
    return S_OK;
}

IFACEMETHODIMP Provider::GetFieldDescriptorCount(DWORD* count) {
    *count = FI_COUNT;
    return S_OK;
}

IFACEMETHODIMP Provider::GetFieldDescriptorAt(DWORD index,
                                              CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR** descriptor) {
    if (index >= FI_COUNT || !descriptor) return E_INVALIDARG;
    auto* out = static_cast<CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR*>(
        CoTaskMemAlloc(sizeof(CREDENTIAL_PROVIDER_FIELD_DESCRIPTOR)));
    if (!out) return E_OUTOFMEMORY;
    *out = kFields[index];
    HRESULT hr = SHStrDupW(kFields[index].pszLabel, &out->pszLabel);
    if (FAILED(hr)) {
        CoTaskMemFree(out);
        return hr;
    }
    *descriptor = out;
    return S_OK;
}

IFACEMETHODIMP Provider::GetCredentialCount(DWORD* count, DWORD* defaultIndex, BOOL* autoLogon) {
    *count = (_enabled && _userPresent) ? 1 : 0;
    *defaultIndex = CREDENTIAL_PROVIDER_NO_DEFAULT;
    *autoLogon = FALSE;
    if (*count && Matched()) {
        *defaultIndex = 0;
        *autoLogon = TRUE;       // LogonUI calls GetSerialization straight away
    }
    return S_OK;
}

IFACEMETHODIMP Provider::GetCredentialAt(DWORD index, ICredentialProviderCredential** credential) {
    if (index != 0 || !credential || !_credential) return E_INVALIDARG;
    return _credential->QueryInterface(IID_PPV_ARGS(credential));
}

std::wstring Provider::Status() {
    AcquireSRWLockShared(&_lock);
    std::wstring s = _status;
    ReleaseSRWLockShared(&_lock);
    return s;
}

bool Provider::Matched() {
    AcquireSRWLockShared(&_lock);
    bool m = _matched;
    ReleaseSRWLockShared(&_lock);
    return m;
}

bool Provider::TakeCredential(std::wstring& domain, std::wstring& user, std::wstring& password) {
    AcquireSRWLockExclusive(&_lock);
    bool ok = _matched;
    if (ok) {
        domain = _domain;
        user = _user;
        password = _password;       // one use only
        WipeString(_password);
        _matched = false;
    }
    ReleaseSRWLockExclusive(&_lock);
    return ok;
}

void Provider::StartScan() {
    AcquireSRWLockExclusive(&_lock);
    bool start = !_scanning && !_matched && _events != nullptr;
    if (start) {
        _scanning = true;
        _cancel = false;
        _status = L"Looking for you\x2026";
    }
    ReleaseSRWLockExclusive(&_lock);
    if (!start) return;

    AddRef();                       // the worker holds the provider (and so the DLL) alive
    HANDLE thread = CreateThread(nullptr, 0, ScanThread, this, 0, nullptr);
    if (thread) {
        CloseHandle(thread);
    } else {
        AcquireSRWLockExclusive(&_lock);
        _scanning = false;
        ReleaseSRWLockExclusive(&_lock);
        Release();
    }
}

DWORD WINAPI Provider::ScanThread(LPVOID param) {
    auto* self = static_cast<Provider*>(param);
    self->Scan();
    self->Release();
    return 0;
}

static HANDLE OpenServicePipe() {
    for (int attempt = 0; attempt < 6; ++attempt) {
        // SECURITY_IDENTIFICATION: whoever owns the pipe cannot act as LogonUI (SYSTEM).
        HANDLE pipe = CreateFileW(kPipeName, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                                  SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
        if (pipe != INVALID_HANDLE_VALUE) {
            DWORD mode = PIPE_READMODE_MESSAGE;
            SetNamedPipeHandleState(pipe, &mode, nullptr, nullptr);
            return pipe;
        }
        if (GetLastError() == ERROR_PIPE_BUSY) {
            WaitNamedPipeW(kPipeName, 1500);
        } else {
            return INVALID_HANDLE_VALUE;        // service not running: stay silent
        }
    }
    return INVALID_HANDLE_VALUE;
}

void Provider::Scan() {
    HANDLE pipe = OpenServicePipe();
    if (pipe == INVALID_HANDLE_VALUE) {
        Finish(false, L"FaceGate is not running");
        return;
    }
    AcquireSRWLockExclusive(&_lock);
    bool cancelled = _cancel;
    if (!cancelled) _pipe = pipe;
    ReleaseSRWLockExclusive(&_lock);

    wchar_t reply[1024];
    DWORD bytes = 0;
    BOOL ok = FALSE;
    if (!cancelled) {
        static const wchar_t request[] = L"AUTH";
        DWORD written = 0;
        ok = WriteFile(pipe, request, sizeof(request) - sizeof(wchar_t), &written, nullptr) &&
             ReadFile(pipe, reply, sizeof(reply) - sizeof(wchar_t), &bytes, nullptr);
    }

    AcquireSRWLockExclusive(&_lock);
    _pipe = INVALID_HANDLE_VALUE;
    ReleaseSRWLockExclusive(&_lock);
    CloseHandle(pipe);

    if (!ok) {
        SecureZeroMemory(reply, sizeof(reply));
        Finish(false, L"");
        return;
    }
    reply[bytes / sizeof(wchar_t)] = L'\0';

    // "OK\tdomain\tuser\tpassword"  or  "FAIL\treason"
    std::wstring parts[4];
    int part = 0;
    for (const wchar_t* p = reply; *p; ++p) {
        if (*p == L'\t' && part < 3) {
            ++part;
        } else {
            parts[part].push_back(*p);
        }
    }
    SecureZeroMemory(reply, sizeof(reply));

    if (parts[0] == L"OK" && part == 3 && !parts[2].empty()) {
        AcquireSRWLockExclusive(&_lock);
        _domain = parts[1];
        _user = parts[2];
        _password = parts[3];
        ReleaseSRWLockExclusive(&_lock);
        WipeString(parts[3]);
        Finish(true, L"Welcome");
    } else {
        WipeString(parts[3]);
        Finish(false, parts[1].empty() ? L"Face not recognised" : parts[1]);
    }
}

void Provider::Finish(bool matched, const std::wstring& status) {
    ICredentialProviderEvents* events = nullptr;
    UINT_PTR context = 0;
    AcquireSRWLockExclusive(&_lock);
    _scanning = false;
    if (_cancel) {
        matched = false;            // the lock screen went away; drop the result
        WipeString(_password);
    }
    _matched = matched;
    if (!status.empty()) _status = status;
    if (!_cancel && _events) {
        events = _events;
        events->AddRef();
        context = _adviseContext;
    }
    ReleaseSRWLockExclusive(&_lock);
    if (events) {
        events->CredentialsChanged(context);    // LogonUI re-enumerates; auto-logon if matched
        events->Release();
    }
}

void Provider::ReportRejected() {
    HANDLE pipe = OpenServicePipe();
    if (pipe == INVALID_HANDLE_VALUE) return;
    static const wchar_t request[] = L"REJECTED";
    DWORD n = 0;
    wchar_t reply[16];
    if (WriteFile(pipe, request, sizeof(request) - sizeof(wchar_t), &n, nullptr))
        ReadFile(pipe, reply, sizeof(reply), &n, nullptr);
    CloseHandle(pipe);
    AcquireSRWLockExclusive(&_lock);
    _status = L"Saved password rejected";
    ReleaseSRWLockExclusive(&_lock);
}

// ---------------------------------------------------------------------------
// Class factory and DLL exports
// ---------------------------------------------------------------------------

class ClassFactory : public IClassFactory {
public:
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IClassFactory) {
            *ppv = static_cast<IClassFactory*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    IFACEMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&_refs); }
    IFACEMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&_refs);
        if (!n) delete this;
        return n;
    }
    IFACEMETHODIMP CreateInstance(IUnknown* outer, REFIID riid, void** ppv) override {
        if (outer) return CLASS_E_NOAGGREGATION;
        Provider* provider = new (std::nothrow) Provider();
        if (!provider) return E_OUTOFMEMORY;
        HRESULT hr = provider->QueryInterface(riid, ppv);
        provider->Release();
        return hr;
    }
    IFACEMETHODIMP LockServer(BOOL lock) override {
        if (lock) DllAddRef(); else DllRelease();
        return S_OK;
    }

private:
    LONG _refs = 1;
};

STDAPI DllGetClassObject(REFCLSID clsid, REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (clsid != CLSID_FaceGateProvider) return CLASS_E_CLASSNOTAVAILABLE;
    ClassFactory* factory = new (std::nothrow) ClassFactory();
    if (!factory) return E_OUTOFMEMORY;
    HRESULT hr = factory->QueryInterface(riid, ppv);
    factory->Release();
    return hr;
}

STDAPI DllCanUnloadNow() { return g_refs > 0 ? S_FALSE : S_OK; }

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_hinst = instance;
        DisableThreadLibraryCalls(instance);
    }
    return TRUE;
}
