import os

# Clean invalid bracketed IPv6 in NO_PROXY/no_proxy which breaks httpx URLPattern parsing
for var in ("NO_PROXY", "no_proxy"):
    val = os.environ.get(var)
    if val and "[::1]" in val:
        os.environ[var] = val.replace("[::1]", "::1")
