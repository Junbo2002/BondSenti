import pandas as pd
from difflib import SequenceMatcher

# ===================
# 消歧模型的阈值
# ===================

CLOSE_DISAMBIGUATOR_THRESHOLD = 0.6
OPEN_DISAMBIGUATOR_THRESHOLD = 0.4

# ===========================
# 实体消歧(基于知识库)
# ===========================

class StringMatchingDisambiguator:
    def __init__(self, data_path="datas/filtered_data.csv"):
        self.df = pd.read_csv(data_path, encoding="utf-8")
        self.entity_set = set(self.df["公司"].values)

    def disambiguate(self, entity_list):
        disambiguated_list = set()
        for entity in entity_list:
            matched_entity, similarity = self.match_entity(entity)
            # print(entity, matched_entity, similarity)
            if similarity > CLOSE_DISAMBIGUATOR_THRESHOLD:
                disambiguated_list.add(matched_entity)
        return list(disambiguated_list)

    def match_entity(self, entity):
        best_match = None
        max_similarity = 0.0
        for candidate in self.entity_set:
            similarity = self.calculate_similarity(entity, candidate)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = candidate
        return best_match, max_similarity

    def calculate_similarity(self, entity1, entity2):
        # 在此处实现字符串相似度度量方法，例如编辑距离、余弦相似度等
        # 返回相似度分值，范围通常在0到1之间
        # TODO 这里可以再优化一下，一是用其他的度量方法（编辑距离）；二是计算相似度时去掉一些无关的词（如“公司”、"集团"）
        return SequenceMatcher(None, entity1, entity2).ratio()


# ===========================
# 实体消歧(基于连通图)
# ===========================
class ConnectedComponentsDisambiguator:
    def __init__(self):
        pass

    def calculate_similarity(self, entity_list):
        similarity_matrix = [[SequenceMatcher(None, entity1, entity2).ratio()
                              for entity1 in entity_list] for entity2 in entity_list]
        return similarity_matrix

    def disambiguate(self, entity_list, threshold=OPEN_DISAMBIGUATOR_THRESHOLD):
        similarity_matrix = self.calculate_similarity(entity_list)
        num_entities = len(similarity_matrix)
        visited = [False] * num_entities
        connected_components = []

        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(num_entities):
                if similarity_matrix[node][neighbor] >= threshold and not visited[neighbor]:
                    dfs(neighbor, component)

        for node in range(num_entities):
            if not visited[node]:
                component = []
                dfs(node, component)
                connected_components.append(component)

        # 从每个连通分支中选择一个实体作为代表
        representatives = self.get_representatives(entity_list, similarity_matrix, connected_components)
        return representatives

    def get_representatives(self, entity_list, similarity_matrix, connected_components):
        representatives = []
        for component in connected_components:
            max_similarity = 0.0
            representative = None
            for entity in component:
                similarity = sum(similarity_matrix[entity])
                if similarity > max_similarity:
                    max_similarity = similarity
                    representative = entity
            representatives.append(entity_list[representative])
        return representatives


if __name__ == '__main__':
    # model = StringMatchingDisambiguator()
    entity_list = ["中国石油", "中国石化", "中石化", "中石油", "中国石油集团", "石油天然气"]
    # print(model.entity_set)
    # print(model.disambiguate(entity_list))

    model = ConnectedComponentsDisambiguator()
    print(model.disambiguate(entity_list))