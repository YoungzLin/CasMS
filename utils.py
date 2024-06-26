from texttable import Texttable
import sys

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def create_numeric_mapping(node_properties):
    return {value:i for i, value in enumerate(node_properties)}
class Logger(object):
    def __init__ (self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write (self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush (self):
        pass
