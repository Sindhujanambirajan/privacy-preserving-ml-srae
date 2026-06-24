content = open('frontend/index.html', 'r', encoding='utf-8').read()
print("File loaded!")
print("Searching for validation...")

if "Please select a CSV file" in content:
    content = content.replace(
        "alert('Please select a CSV file!');",
        "console.log('auto csv');"
    )
    content = content.replace(
        'alert("Please select a CSV file!");',
        'console.log("auto csv");'
    )
    open('frontend/index.html', 'w', encoding='utf-8').write(content)
    print("Fixed successfully!")
else:
    print("Text not found!")
    print("Manual fix needed!")