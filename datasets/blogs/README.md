Several real-world network datasets with binary node attributes. For each dataset, we extract the largest connected component, remove nodes without attribute information, and eliminate self-loops.

| **Dataset**                   | **Nodes** | **Edges** | **Protected Attribute(s)** | **Description**                                                                                                                                 | **Source**                                                                                                                                                               |
| ----------------------------- | --------: | --------: | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Political Blogs**           |      1222 |     16714 | Political leaning          | Directed hyperlinks between US political blogs (Feb 2005). Labels: 0 = liberal, 1 = conservative.                                               | [Link](https://websites.umich.edu/~mejn/netdata/)                                                                                  |
| **Facebook Net**              |       155 |      1412 | Gender                     | High-school Facebook friendship network (Marseilles, 2013). Edge weight 1 = friendship, 0 = no friendship.                                      | [Link](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) |
| **Books**                     |        92 |       374 | Political leaning          | US political book co-purchase network. Labels: liberal or conservative (neutral removed).                                                       | [Link](https://websites.umich.edu/~mejn/netdata/)                                                                                  |
| **Twitter Political Retweet** |     18470 |     48053 | Political leaning          | Directed retweet network (political). Labels: 1 = group A, 0 = group B.                                                                         | [Link](https://networkrepository.com/)                                                                                                         |
| **Drug Net**                  |       185 |       265 | Gender                     | Directed acquaintanceship network among Hartford drug users from ethnographic study.                                                            | [Link](https://sites.google.com/site/ucinetsoftware/datasets/covert-networks/drugnet)           |
| **Friendship Net**            |       127 |       396 | Gender                     | Directed reported friendships in a high school (Marseilles, 2013).                                                                              | [Link](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) |
| **Facebook Ego**              |      4039 |     88234 | Gender                     | Ego-networks with circles and node features from Facebook survey participants.                                                                  | [Link](https://snap.stanford.edu/data/ego-Facebook.html)                                                                                                                 |
| **Deezer Europe**             |     28281 |     92752 | Gender                     | Music social network (Europe). Node attributes include gender.                                                                                  | [Link](https://snap.stanford.edu/data/feather-deezer-social.html)                                                                                                        |



| Synthetic Datasets | Description | Code|
| :--- | :--- | :--- |
| **Variant of Stochastic Block Model** | A variant of the Stochastic Block Model that incorporates fairness constraints. Introduced by Kleindessner et al. in [*Guarantees for Spectral Clustering with Fairness Constraints*](https://proceedings.mlr.press/v97/kleindessner19b/kleindessner19b.pdf). | [Link](graphs/v_sbm.py)


Real-world network datasets with binary node attributes.
For each dataset, we extract the largest connected component, remove all nodes lacking attribute information, and eliminate self-loop edges.
Each dataset is provided in two files:

edges.txt — list of edges in the network
attributes.txt — binary attribute values for each node