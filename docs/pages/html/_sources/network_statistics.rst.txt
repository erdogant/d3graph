Network Statistics
######################

In networks, not all nodes contribute equally to the structure and information flow. Some nodes play a more central role because they connect many entities, bridge different communities, or represent important sources of information. Identifying these key nodes is one of the key tasks in network analysis and can provide valuable insights. In a social network, important nodes may represent influential users, trusted sources, or individuals who connect different groups. In a personal knowledge network, important nodes can reveal central concepts, frequently referenced documents, or ideas that form the foundation of your knowledge structure. However, importance is not a single concept; a node can be important because it has many connections, because it connects otherwise separated parts of the network, or because it is linked to other influential nodes.


PageRank
************************************

PageRank identifies nodes that are important based on the importance of their connected neighbors. Unlike simple counting of connections, PageRank considers the quality of those connections: a link from an influential node contributes more than a link from a less important node. In social network analysis, PageRank can identify influential users, opinion leaders, or accounts that have a strong position in information flow. A user does not necessarily need many connections; being connected to other influential users can also increase their ranking. In a personal knowledge network, PageRank can reveal the most central concepts, documents, or notes in your knowledge base. These are the topics that are strongly connected to other important ideas and may represent the core themes of your personal knowledge system.

.. code:: python
	
    import pandas as pd
    from d3blocks import D3Blocks
    
    # PageRank-focused
    # Many nodes point to A with strong weights → A has highest PageRank
    df = pd.DataFrame({
        'source': ['B','C','D','E','F','G','H','I'],
        'target': ['A','A','A','A','A','A','A','A'],
        'weight': [1,2,3,4,5,2,1,3]
    })
    
    # Most important node for PageRank: A
    d3 = D3Blocks()
    d3.d3graph(df, filepath='pagerank_graph.html', directed=True, showfig=True)


HITS (Hub)
************************************

The HITS Hub score identifies nodes that act as strong collectors of information. A high hub score indicates that a node points towards many authoritative nodes. Hubs are typically not the source of information themselves but provide valuable connections to important resources. In social networks, high hub scores can identify users who actively share or reference influential accounts, acting as information aggregators. In a knowledge graph, hubs can represent overview notes, indexes, or personal dashboards that connect many important concepts. For example, a project overview document linking to research papers, meeting notes, and technical documentation may naturally become a high-scoring hub.

.. code:: python
	
    import pandas as pd
    from d3blocks import D3Blocks
    
    # HITS-focused graph
    # Hubs (H1,H2) → Authorities (A1,A2,A3)
    df = pd.DataFrame({
        'source': ['H1','H1','H1','H2','H2','H2'],
        'target': ['A1','A2','A3','A1','A2','A3'],
        'weight': [3,2,1,3,2,1]
    })
    
    # Most important nodes:
    #   Authorities: A1, A2, A3
    #   Hubs: H1, H2
    
    d3 = D3Blocks()
    d3.d3graph(df, filepath='hits_graph.html')



HITS (Authority)
************************************

The HITS Authority score identifies nodes that contain valuable information based on the number and quality of incoming connections. A high authority score means that many important hubs reference this node. In social network analysis, authority scores can identify users whose content is frequently referenced or shared by influential accounts. These users may represent experts or trusted sources. In a personal knowledge graph, authority scores can reveal the most valuable documents, notes, or concepts in your collection. For example, a technical document that is repeatedly referenced by project notes, research summaries, and related concepts becomes an authoritative source within your personal knowledge system.


Degree Centrality
************************************

Degree centrality measures the number of direct connections a node has. It is one of the simplest network statistics and provides an immediate indication of how connected an entity is within the network. In social network analysis, degree centrality identifies highly connected individuals, such as users with many followers or many interactions. These nodes often represent popular or highly active participants. In a second brain network, degree centrality can reveal frequently referenced notes or concepts. A document connected to many other notes may represent a central topic, a frequently used resource, or a key area of interest. However, degree alone does not indicate importance because many connections may come from less relevant nodes.

.. code:: python

    import pandas as pd
    from d3blocks import D3Blocks
    
    # Degree-focused graph
    # X connects to many nodes with strong weights → highest degree centrality
    df = pd.DataFrame({
        'source': ['X','X','X','X','X','A','B','C'],
        'target': ['A','B','C','D','E','F','G','H'],
        'weight': [5,4,3,2,1,1,1,1]
    })
    
    # Most important node for Degree Centrality: X
    
    d3 = D3Blocks()
    d3.d3graph(df, filepath='degree_graph.html')




Closeness Centrality
************************************

Closeness centrality measures how easily a node can reach all other nodes in the network. Nodes with high closeness values are positioned near the center of the network and can efficiently spread or access information. In social networks, closeness centrality can identify users who are well positioned to communicate information throughout the network. These users may act as efficient information spreaders because they have relatively short paths to many others. In a personal knowledge graph, closeness can reveal concepts that connect different areas of knowledge. A note with high closeness may represent a bridge between multiple topics, allowing you to navigate quickly from one part of your knowledge base to another.

.. code:: python

    import pandas as pd
    from d3blocks import D3Blocks
    
    # Closeness-focused graph
    # C is the center of a star → shortest paths to all nodes → highest closeness
    df = pd.DataFrame({
        'source': ['C','C','C','C','C'],
        'target': ['A','B','D','E','F'],
        'weight': [1,1,1,1,1]
    })
    
    # Most important node for Closeness Centrality: C
    
    d3 = D3Blocks()
    d3.d3graph(df, filepath='closeness_graph.html')





Betweenness Centrality
************************************

Betweenness centrality identifies nodes that frequently appear on the shortest paths between other nodes. These nodes often act as bridges connecting different parts of a network. In social network analysis, high betweenness nodes can represent influential connectors, such as people who link different communities or departments. Removing these nodes may significantly affect information flow through the network. In a second brain network, betweenness centrality can identify concepts that connect otherwise separate knowledge areas. For example, a note about "machine learning" may connect mathematics, programming, statistics, and artificial intelligence topics, acting as a bridge between different knowledge domains.

.. code:: python

    import pandas as pd
    from d3blocks import D3Blocks
    
    # Betweenness-focused graph
    # M is the only connector between two clusters → highest betweenness
    df = pd.DataFrame({
        'source': ['A','B','C','M','M','X','Y','Z'],
        'target': ['M','M','M','X','Y','M','M','M'],
        'weight': [1,1,1,3,3,1,1,1]
    })
    
    # Most important node for Betweenness Centrality: M
    
    d3 = D3Blocks()
    d3.d3graph(df, filepath='betweenness_graph.html')




Network Clustering
######################

Network clustering identifies groups of nodes that are more strongly connected with each other than with the rest of the network. These groups represent communities or natural structures that emerge from the relationships in the data. In social network analysis, clustering can reveal groups of friends, professional communities, or interest-based communities without requiring predefined labels. In a personal knowledge graph, clustering can automatically discover knowledge domains within your notes and documents. For example, a large collection of personal files may naturally separate into clusters such as programming, research, travel, finance, or hobbies. This provides a data-driven view of how your own knowledge is organized and can reveal unexpected relationships between topics.




Statistical Testing
######################

Distinguishing meaningful structures from random patterns is one of the fundamental challenges in network analysis. A highly connected node or dense cluster may appear important, but these patterns can also arise purely by chance. To determine whether an observed network property is statistically significant, the network can be compared against a collection of randomized networks that preserve key characteristics, such as the degree distribution. By generating a null distribution for network statistics such as PageRank, HITS scores, or clustering coefficients, it becomes possible to assign statistical significance to the observed values. Within D3graph, these significant nodes or structures can subsequently be highlighted in the interactive visualization, allowing users to focus on relationships that are unlikely to have occurred by random chance and are therefore more likely to represent meaningful underlying organization.
To create a null distribution for a network, we can repeatedly generate randomized versions of the original network while preserving one or more important structural properties. The most common approach is degree-preserving randomization, in which edges are randomly rewired but each node keeps the same number of connections (its degree).


.. code:: python

    from d3graph import d3graph, vec2adjmat
    
    # Initialize
    d3 = d3graph()
    
    # Load example data
    df = d3.import_example('socialmedia')
    
    # Slice first 1000 rows for demonstration purposes
    df = df[0:2000]
    
    # Create adjmat
    adjmat = vec2adjmat(source=df['source'], target=df['target'], weight=df['weight'])
    
    # Create graph
    d3.graph(adjmat)
    
    # Compute node significance for specified network statistic
    d3.network_significance(adjmat, 'pagerank', n_top=100, n_random=1000)
    
    # Show graph with custom specific settings
    d3.show()


.. include:: add_bottom.add