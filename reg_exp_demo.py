import re
import sys

# read input
path = sys.argv[1]
with open(path, 'r') as f:
    input = f.read()

    # use a modified regex string because Star's didn't work. Desired behavior is unkown, but this works for the demo
    pattern = re.compile(r"\$\$((\w+[,'|‘’]\s)+)\(((\w+[,'|‘’]\s)+)\)")
    repl = r'<span class="tooltip">\1<span class="tooltiptext">\(\3\) </span> </span>'

    # do the actual regex capture group string replacement
    output = pattern.sub(repl, input)

    # save output
    output_file = open('hoverable_html.html', 'w')
    output_file.write(output)
    output_file.close()