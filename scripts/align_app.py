from flask import Flask, render_template_string, request
from viz_align import viz_sa_input

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_dims = list(map(int, request.form['input_dims'].split()))
        sa_arch = list(map(int, request.form['sa_arch'].split()))
        dram_width = int(request.form['dram_width'])
        table_data = viz_sa_input(input_dims, sa_arch, dram_width)
        return render_template_string(template, table_data=table_data, 
                                      input_dims=request.form['input_dims'],
                                      sa_arch=request.form['sa_arch'],
                                      dram_width=request.form['dram_width'])
    return render_template_string(template, table_data=None, 
                                  input_dims='5 3 5', sa_arch='9 8 8', dram_width='32')

template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization of Aligned Dimensions</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 8px;
            text-align: center;
        }
        input {
            padding: 5px;
            margin: 5px;
        }
    </style>
</head>
<body>

<h2>Enter the parameters</h2>
<form method="POST">
    <label for="input_dims">Input Dimensions (depth height width):</label><br>
    <input type="text" id="input_dims" name="input_dims" value="{{ input_dims }}"><br><br>
    
    <label for="sa_arch">Architecture (depth height width):</label><br>
    <input type="text" id="sa_arch" name="sa_arch" value="{{ sa_arch }}"><br><br>
    
    <label for="dram_width">DRAM Width:</label><br>
    <input type="text" id="dram_width" name="dram_width" value="{{ dram_width }}"><br><br>
    
    <input type="submit" value="Submit">
</form>

{% if table_data %}
    <h3>Output Table:</h3>
    <table>
        <tr>
            <th>Index</th>
            <th>Values</th>
        </tr>
        {% for row in table_data %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ row | join(", ") }}</td>
        </tr>
        {% endfor %}
    </table>
{% endif %}

</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
