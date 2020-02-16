import hashlib
import json


class Headline(object):
    headline = ""
    link = ""
    semantic_value = 0.0
    origin = ""
    datetime = ""
    neg = 0.0
    pos = 0.0
    neu = 0.0
    predicted_class = 0
    display_image = ""

    def __str__(self, *args, **kwargs):
        return "{: <120} {: <20} {: <10} {: <25} {: <180} {: <25} {: <180}".format(self.headline, str(self.predicted_class),
                                                                                   str(
            self.semantic_value), self.origin,
            self.link, str(self.datetime), self.display_image or "")

    def __hash__(self, *args, **kwargs):

        string = "{: <120} {: <10} {: <25} {: <180} {: <10} {: <10} {: <10}".format(self.headline,
                                                                                    str(
                                                                                        self.semantic_value),
                                                                                    self.origin, self.link,
                                                                                    str(self.pos), str(
                                                                                        self.neg),
                                                                                    str(self.neu))

        return hash(string)

    def sha256(self):
        string = "{: <120} {: <10} {: <25} {: <180} {: <10} {: <10} {: <10}".format(self.headline,
                                                                                    str(
                                                                                        self.semantic_value),
                                                                                    self.origin, self.link,
                                                                                    str(self.pos), str(
                                                                                        self.neg),
                                                                                    str(self.neu))

        hash_object = hashlib.sha256(bytes(string, 'utf-8'))
        hex_dig = hash_object.hexdigest()

        return hex_dig

    # FROM: http://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)

    def to_array(self):
        return [self.headline, self.origin, self.semantic_value, self.pos, self.neg, self.neu, self.datetime]

    def __init__(self, headline, link, origin, datetime, display_image):
        self.headline = headline
        self.link = link
        self.origin = origin
        self.datetime = datetime
        self.display_image = display_image
