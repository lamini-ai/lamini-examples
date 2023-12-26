import gzip
import jsonlines

def main():
    entries = load_entries()

    descriptions = generate_descriptions(entries)

    save_descriptions(descriptions)

def load_entries():

    with gzip.open('best_entities.jsonl.gz', 'rt') as f:
        reader = jsonlines.Reader(f)

        for entry in reader:
            yield entry

def generate_descriptions(entries):
    for entry in entries:
        description = ""
        if 'code' in entry and len(entry['code']) > 0:
            description = "Code: " + entry['code'] + "\n"

        if 'title' in entry:
            description += "Title: " + entry['title']['@value'] + "\n"

        if 'definition' in entry:
            description += "Definition: " + entry['definition']['@value'] + "\n"

        if 'longDefinition' in entry:
            description += "Long Definition: " + entry['longDefinition']['@value'] + "\n"

        entry['description'] = description

        yield entry

def save_descriptions(entries):
    with gzip.open('best_entities_with_descriptions.jsonl.gz', 'wt') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(entries)

main()



