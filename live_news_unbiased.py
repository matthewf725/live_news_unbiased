import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords, words, names
import re
import random
import matplotlib.colors as mcolors

def scrape_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def extract_links(soup):
    return [a['href'] for a in soup.find_all('a', href=True)]

def scrape_and_process(url, site_title):
    soup = scrape_page(url)
    text = soup.get_text()
    links = extract_links(soup)
    words = clean_and_tokenize(text, site_title)
    return words, links

def clean_and_tokenize(text, site_title):
    stop_words = set(stopwords.words('english'))
    english_words = set(words.words())
    english_names = set(names.words())
    valid_words = english_words | english_names
    site_title = site_title.lower()
    words_list = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words_list if word not in stop_words and word != site_title and word in valid_words]
    return filtered_words

def fetch_and_process_data(urls):
    all_data = {}
    
    for entry in urls:
        url = entry['url']
        title = entry['title']
        words, links = scrape_and_process(url, title)
        all_data[url] = {
            'words': words,
            'links': links
        }
    
    return all_data

def assign_colors(data):
    color_map = {}
    colors = list(mcolors.TABLEAU_COLORS.values())
    random.shuffle(colors)
    for i, url in enumerate(data.keys()):
        color_map[url] = colors[i % len(colors)]
    return color_map

def create_network_graph(data, color_map, top_n=50):
    G = nx.Graph()
    word_link_map = defaultdict(set)
    
    for main_url, content in data.items():
        main_word_counts = Counter(content['words']).most_common(top_n)
        main_words = [word for word, _ in main_word_counts]
        
        for word in main_words:
            word_link_map[word].add(main_url)
        
        for word, count in main_word_counts:
            G.add_node(word, size=count)
        
        for i, word1 in enumerate(main_words):
            for word2 in main_words[i+1:]:
                if word1 != word2:
                    G.add_edge(word1, word2, weight=1)
        
        for link in content['links']:
            if link not in G:
                G.add_node(link, size=1, color=color_map.get(link, 'gray'))
            G.add_edge(main_url, link, weight=0.5)
    
    for word, links in word_link_map.items():
        for link in links:
            G.add_edge(word, link, weight=0.5)
    
    return G

def draw_network_graph(G, ax):
    ax.clear()
    ax.set_facecolor('black')
    pos = nx.spring_layout(G, k=0.15, iterations=50)
    sizes = [G.nodes[node].get('size', 1) * 5 for node in G.nodes]
    max_connectivity = max(len(G[node]) for node in G.nodes)
    node_colors = []
    edge_colors = []
    for node in G.nodes:
        if isinstance(node, str) and node.startswith(('https://', 'http://')):
            node_colors.append('gray')
        else:
            connectivity = len(G[node])
            node_colors.append(plt.cm.viridis(connectivity / max_connectivity))
    
    cmap = plt.get_cmap('viridis')
    for u, v, data in G.edges(data=True):
        if u in G and v in G:
            connectivity_u = len(G[u])
            connectivity_v = len(G[v])
            edge_color = cmap((connectivity_u + connectivity_v) / (2 * max_connectivity))
            edge_colors.append(edge_color)
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color=edge_colors)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, ax=ax)
    
    max_font_size = 11
    for node, (x, y) in pos.items():
        size_index = list(G.nodes).index(node)
        circle_radius = sizes[size_index] * 0.001
        ftsize = min(sizes) * 0.001
        font_size = ftsize * 100
        ax.text(x, y, node, fontsize=font_size, ha='center', va='center', color='white', zorder=3)
    
    ax.set_title('Webpage Word Network', color='white')
    ax.axis('off')

def main():
    nltk.download('stopwords')
    nltk.download('words')
    nltk.download('names')
    
    urls = [
        {'url': 'https://www.bbc.com/news', 'title': 'bbc'},
        {'url': 'https://www.cnn.com', 'title': 'cnn'},
        {'url': 'https://www.reuters.com', 'title': 'reuters'},
        {'url': 'https://www.nytimes.com', 'title': 'nytimes'},
        {'url': 'https://www.theguardian.com', 'title': 'guardian'},
        {'url': 'https://www.washingtonpost.com', 'title': 'washingtonpost'},
        {'url': 'https://www.aljazeera.com', 'title': 'aljazeera'},
        {'url': 'https://www.nbcnews.com', 'title': 'nbcnews'},
        {'url': 'https://www.foxnews.com', 'title': 'foxnews'},
        {'url': 'https://www.bloomberg.com', 'title': 'bloomberg'}
    ]
    
    data = fetch_and_process_data(urls)
    color_map = assign_colors(data)
    G = create_network_graph(data, color_map)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('black')
    draw_network_graph(G, ax)
    plt.show()

if __name__ == "__main__":
    main()
