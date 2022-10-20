from jinja2 import FileSystemLoader, Environment


def generate_html(n_sec=4, n_rec=9):
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    template = env.get_template("mos.html.jinja2")

    sections = []
    for i in range(n_sec):
        section = {
            "id": i+1,
            "recordings": [],
        }
        for j in range(n_rec):
            recording = {
                "id": j+1,
            }
            section["recordings"].append(recording)
        sections.append(section)

    html = template.render(
        sections=sections,
    )
    return html

if __name__ == "__main__":
    html = generate_html(2, 9)
    print(html)
