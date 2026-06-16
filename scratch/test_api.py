import urllib.request
import json

api_key = "sk-hPx6ULypAJT5gS4OtBgWhFF97zSB8i1BlHa50wTvTKsluTlH"
base_url = "https://api.shopaikey.com/v1"

def test_models():
    req = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("Models:")
            for m in data.get("data", []):
                print(f"- {m.get('id')}")
    except Exception as e:
        print(f"Error listing models: {e}")

test_models()
