import json

def txt_to_json(txt_file, json_file):
    data = {}
    
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove any whitespace or newline characters from the line
            line = line.strip()
            if line:
                # Split line into key and value based on ':' separator
                print(line)
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

    # Write the dictionary to a JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
txt_file = '../scripts/results.txt'  # Replace with your .txt file
json_file = '../eval/temp.json'  # Name for the output .json file
txt_to_json(txt_file, json_file)
print(f"Converted {txt_file} to {json_file}")
