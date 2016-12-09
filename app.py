# communication with Ozono platform
from suds.client import Client
import json
import sys

# Generate the server of web services
import logging

logger = logging.getLogger(__name__)

from spyne.application import Application

from spyne.decorator import rpc, srpc

from spyne.protocol.soap import Soap11
from spyne.protocol.soap import Soap12
from spyne.service import ServiceBase
from spyne.protocol.http import HttpRpc
from spyne.error import ResourceNotFoundError
from spyne.model.primitive import String, Int
from spyne.model.complex import ComplexModel, Array, Iterable, ComplexModelBase
from spyne.protocol.xml import XmlDocument
from spyne.protocol.json import JsonDocument
from spyne.server.wsgi import WsgiApplication
from server import Nltk, Clusterization
import langid

iso639_1_to_native = {
    # 'cs': 'czech', // not supported in nltk stopwords
    'da': 'danish',
    # 'nl': 'dutch', // not supported in nltk stopwords
    'fr': 'french',
    # 'de': 'german', // not supported in nltk stopwords
    'no': 'norwegian',
    # 'pl': 'polish', // not supported in nltk stopwords
    'en': 'english',
    'fi': 'finnish',
    'it': 'italian',
    'es': 'spanish',
    'pt': 'portuguese',
    'sv': 'swedish',
    # 'tr': 'turkish'
}

iso639_1_to_2 = {
    # 'cs': 'cze', // not supported
    'da': 'dan',
    # 'nl': None, // not supported
    'fr': 'fra',
    # 'de': None, // not supported
    'no': 'nno',
    # 'pl': 'pol', // not supported
    'en': 'eng',
    'fi': 'fin',
    'it': 'ita',
    'es': 'spa',
    'pt': 'por',
    'sv': 'swe',
    # 'tr': 'tur'
}


class Message(ComplexModel):
    _type_info = [
        ('msgID', String),
        ('from', String),
        ('message', String),
        ('date', String),
        ('source', String),
        ('link', String),
        ('classification', String),
        ('language', String),
        ('clusterNumber', Int),
        ('clusterCount', Int),
    ]


class ArrayOfMessage(ComplexModel):
    __type_name__ = 'ArrayOfMessage'
    Message = Message.customize(max_occurs='unbounded')


class Filter(ComplexModel):
    __type_name__ = 'Filter'
    __class__ = "f"
    _type_info = {
        'keyword': String(min_occurs='0', max_occurs='1', nullable=False),
        'dateBegin': String(min_occurs='0', max_occurs='1', nullable=False),
        'dateEnd': String(min_occurs='0', max_occurs='1', nullable=False),
        'source': String(min_occurs='0', max_occurs='1', nullable=False),
        'msgCount': Int(min_occurs='1', max_occurs='1', nullable=False),
        'validated': String(min_occurs='0', max_occurs='1', nullable=False),
        'target': String(min_occurs='0', max_occurs='1', nullable=False),
        'status': String(min_occurs='0', max_occurs='1', nullable=False),
        'messageType': String(min_occurs='0', max_occurs='1', nullable=False),
    }


# list of messages
message_list = []
ms_database = []

client = Client('http://ops.tekever.com/webservices/socialmedia/SocialNetworksService.asmx?WSDL')


def get_messages(filter):
    return client.service.recvSocialMediaMessages(filter)[0]


class Sentiment():
    def __init__(self):
        self.nltk = Nltk()
        self.clusterization = Clusterization()

    def compute_all_sentiment(self, filters=None):
        messages = get_messages(filters)
        unclassified_messages = list([m for m in messages
                                      if not hasattr(m, 'classification') and
                                      hasattr(m, 'message') and m.message is not None])

        for m in unclassified_messages:
            scores = self.compute_sentiment(m)
            if scores is not None:
                client.service.updateSocialMediaMessageClassification(m.msgID, json.dumps(scores))

        client.service.sendSocialMediaMessages(messages)

        # make clusterization
        messages_to_cluster = list([m.message.encode('utf-8') for m in messages if hasattr(m, 'message')])  # this will cluster all messages, not only unclassified ones
        lang_of_m_to_cluster = list([iso639_1_to_native.get(m.language, 'english') for m in messages if hasattr(m, 'language')])
        num_clusters_, cluster_labels = self.clusterization.make_clusterization(messages_to_cluster, lang_of_m_to_cluster)  # returns i) the number of identified clusters, and ii) the list of cluster labels (the list has one element for each message)
        print ("\nNumber of identified clusters is= {}", num_clusters_)
        # print "Exit labels is=", labels

        # Print outputs of MS2A analysis: classification and clustering
        # labels_index=0
        for m, labels_index in zip(messages, range(len(cluster_labels))):
            data = {}
            if hasattr(m, 'classification'):
                if m.msgID:
                    data['msgID'] = m.msgID
                if hasattr(m, 'message'):
                    data['message'] = m.message.encode('utf-8')
                if m.date:
                    data['date'] = m.date
                if hasattr(m, 'source'):
                    data['source'] = m.source
                if hasattr(m, 'link'):
                    data['link'] = m.link
                if hasattr(m, 'classification'):
                    data['classification'] = m.classification
                if hasattr(m, 'language'):
                    data['language'] = m.language
                if getattr(m, 'from', 'm.from'):
                    data['from'] = getattr(m, 'from', 'm.from')

                data['clusterCount'] = num_clusters_
                data['clusterNumber'] = cluster_labels[labels_index]

            message_list.append(data)

        return message_list

    def compute_sentiment(self, message):
        if message.language == 'en':  # helps preventing mis-identified english language
            message.language = langid.classify(message.message)[0]

        native_language = iso639_1_to_native.get(message.language, 'english') \
            if hasattr(message, 'language') else 'english'
        iso_639_2 = iso639_1_to_2.get(message.language, 'eng') \
            if hasattr(message, 'language') else 'eng'

        words = self.nltk.tweet_tokenize(message.message.encode('utf8'), True, True, True)
        # print "native language is", native_language
        no_stopwords = self.nltk.remove_stopwords(words, native_language)
        word_sentiments = [senti for senti in
                           map(lambda word: self.compute_word_sentiment(word, iso_639_2), no_stopwords)
                           if senti is not None]

        return {
            'pos': round(sum(map(lambda s: s['pos'], word_sentiments)) / len(word_sentiments), 3),
            'neg': round(sum(map(lambda s: s['neg'], word_sentiments)) / len(word_sentiments), 3),
            'obj': round(sum(map(lambda s: s['obj'], word_sentiments)) / len(word_sentiments), 3)
        } if any(word_sentiments) else None

    def compute_word_sentiment(self, word, lang):
        word_synsets = self.nltk.synsets(word, None, lang)
        word_sentiments = [senti for senti in
                           map(lambda synset: self.nltk.senti_synset(synset), word_synsets)
                           if senti is not None]

        return {
            'pos': sum(map(lambda s: s.pos_score(), word_sentiments)) / len(word_sentiments),
            'neg': sum(map(lambda s: s.neg_score(), word_sentiments)) / len(word_sentiments),
            'obj': sum(map(lambda s: s.obj_score(), word_sentiments)) / len(word_sentiments)
        } if any(word_sentiments) else None


def addMessageInformation(msg):
    #print "EE"
    global ms_database
    ms_database.append(msg)

def addArrayInformation(array):
    global ms_database
    ms_database.append(array)


class SoteriaService(ServiceBase):
    __service_name__ = "SoteriaSendData"

    @srpc(Message,_returns=Message)
    def addMessageInformation(msg):
        addMessageInformation(msg)
        return msg

    @srpc(ArrayOfMessage,_returns=ArrayOfMessage)
    def addArrayInformation(array):
        global ms_database
        addArrayInformation(array)
        return array

    @srpc(_returns=ArrayOfMessage)
    def getSoteriaInformation():
        print "\n---------------------------"
        print "| Ding! Incoming request! |"
        print "---------------------------\n"
        #global ms_database

        return ms_database[0]



application = Application([SoteriaService], tns='http://localhost:8000',
                          in_protocol=Soap11(),
                          out_protocol=Soap11()
                          )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("running web app")
    else:
        print("executing script")
        try:
            from wsgiref.simple_server import make_server
        except ImportError:
            print("Error: example server code requires Python >= 2.5")

        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)
        logging.getLogger('spyne.interface.wsdl11').setLevel(logging.DEBUG)
        wsgi_app = WsgiApplication(application)
        wsgi_app.doc.wsdl11.build_interface_document("http://localhost:8000/Soteria")
        server = make_server('127.0.0.1', 8000, wsgi_app)

        print("listening to http://localhost:8000")
        print("wsdl is at: http://localhost:8000/Soteria?wsdl")

        server.serve_forever()


