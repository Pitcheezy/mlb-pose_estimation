from roboflow import Roboflow

API_KEY = "Bubuom4F7MItqMMTXSxz"

try:
    rf = Roboflow(api_key=API_KEY)
    print("API key is valid!")

    # Try to list available workspaces (if method exists)
    try:
        workspaces = rf.list_workspaces()
        print("\nAvailable workspaces:")
        for ws in workspaces:
            print(f"- {ws}")
    except AttributeError:
        print("\nCould not list workspaces automatically.")
        print("Please check your Roboflow dashboard: https://app.roboflow.com")
        print("Look for your workspace name in the URL or project settings.")

except Exception as e:
    print(f"API key error: {e}")
    print("\nTroubleshooting:")
    print("1. Go to: https://app.roboflow.com/settings/api")
    print("2. Copy your Private API Key")
    print("3. Make sure there are no extra spaces")
