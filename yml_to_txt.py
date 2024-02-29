import yaml

# Open the YAML file and load its content
with open('environment.yml', 'r') as stream:
    env = yaml.safe_load(stream)

# Extract package names and versions
with open('requirements.txt', 'w') as req:
    # Iterate through dependencies
    for dep in env['dependencies']:
        # Split package string into name and version
        if isinstance(dep, str):  # Check if the dependency is a string
            parts = dep.split('=')
            if len(parts) >= 2:  # Check if there's at least name and version
                name, version = parts[0], parts[1]
                req.write(f"{name}=={version}\n")
