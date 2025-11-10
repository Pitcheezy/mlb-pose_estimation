from roboflow import Roboflow

API_KEY = "Bubuom4F7MItqMMTXSxz"

try:
    rf = Roboflow(api_key=API_KEY)
    workspaces = rf.workspaces()
    print("Available workspaces:")
    for workspace in workspaces:
        print(f"- {workspace.name} (URL: {workspace.url})")

except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure your API key is correct!")
