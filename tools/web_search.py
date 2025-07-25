from langchain_community.tools import DuckDuckGoSearchRun

def web_search(query: str):
    search_results_str = DuckDuckGoSearchRun().run(query)
    # Parse the string output into a list of dictionaries
    parsed_results = []
    # Each result is typically formatted as "Title: [title]\nLink: [link]\nSnippet: [snippet]\n\n"
    # We can split by "\n\n" to get individual results, then parse each one.
    individual_results = search_results_str.strip().split('\n\n')
    for res_str in individual_results:
        title = "N/A"
        link = "N/A"
        snippet = "N/A"
        lines = res_str.split('\n')
        for line in lines:
            if line.startswith("Title: "):
                title = line[len("Title: "):]
            elif line.startswith("Link: "):
                link = line[len("Link: "):]
            elif line.startswith("Snippet: "):
                snippet = line[len("Snippet: "):]
        if title != "N/A" or link != "N/A" or snippet != "N/A":
            parsed_results.append({"title": title, "link": link, "snippet": snippet})
    return parsed_results