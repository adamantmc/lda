import operator
import math

class TopicIndex:
    def __init__(self, document_topics):
        self.index = {}
        self.document_norms = {}

        for doc in document_topics:
            self.document_norms[doc] = 0

            for topic in document_topics[doc]:
                #Each topic is a tuple: (topic_id, probability)
                if topic[0] not in self.index:
                    self.index[topic[0]] = {}

                self.index[topic[0]][doc] = topic[1]

                self.document_norms[doc] += topic[1]*topic[1]

            self.document_norms[doc] = math.sqrt(self.document_norms[doc])

        self.document_topics = document_topics

    def query(self, query_doc_topics):
        scores = {}

        for doc in self.document_topics:
            scores[doc] = 0

        norm = 0

        for topic in query_doc_topics:
            norm += topic[1]*topic[1]
            if topic[0] in self.index:
                for doc in self.index[topic[0]]:
                    scores[doc] += topic[1]*self.index[topic[0]][doc]

        norm = math.sqrt(norm)

        for doc in scores:
            scores[doc] = scores[doc] / self.document_norms[doc]*norm

        scores_sorted = sorted(scores.items(), key = operator.itemgetter(1), reverse = True)

        return scores_sorted
