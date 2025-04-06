"""
Test Flask templates
"""

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    """Test route"""
    return render_template('test.html')

if __name__ == '__main__':
    # Create a test template
    with open('templates/test.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Test Template</title>
</head>
<body>
    <h1>Test Template</h1>
    <p>If you can see this, the template system is working correctly.</p>
</body>
</html>
        """)
    
    print("Running test Flask app...")
    app.run(debug=True)
