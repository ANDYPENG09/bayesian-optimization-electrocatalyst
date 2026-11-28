import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from copilot import ImaClient,ImaError,response_data,packet

class MockClient(ImaClient):
    def __init__(self, responses):
        self.responses=iter(responses)
        self.calls=[]
    def post(self, action,payload):
        self.calls.append((action,payload))
        return next(self.responses)

class TestIma(unittest.TestCase):
    def test_response_formats(self):
        self.assertEqual(response_data({"retcode":0,"data":{"x":1}}),{"x":1})
        self.assertEqual(response_data({"code":0,"data":{"x":1}}),{"x":1})
        with self.assertRaises(ImaError):
            response_data({"retcode":220030,"data":{}})
    def test_pagination_and_discovery(self):
        c=MockClient([
            {"info_list":[{"media_id":"a","title":"Paper A","highlight_content":"<em>IrOx</em> evidence"},{"media_id":"b","title":"Paper B"}],"next_cursor":"next"},
            {"info_list":[{"media_id":"a","title":"duplicate"},{"media_id":"c","title":"Paper C","highlight_content":"OER"}],"is_end":True}
        ])
        result=packet("IrOx",c.search("kb","IrOx",10),"ima")
        self.assertEqual(len(result["evidence"]),2)
        self.assertEqual(len(result["discovery_only"]),1)
        self.assertIsNone(result["evidence"][0]["page"])
        self.assertEqual(result["evidence"][0]["body"],"IrOx evidence")
        self.assertEqual(c.calls[1][1]["cursor"],"next")
    def test_repeated_cursor(self):
        c=MockClient([{"info_list":[],"next_cursor":"next"},{"info_list":[],"next_cursor":"next"}])
        with self.assertRaises(ImaError):
            c.libraries()
    def test_no_snippet_is_not_evidence(self):
        self.assertEqual(packet("x",[{"title":"x","body":""}],"ima")["evidence"],[])
if __name__=="__main__":unittest.main()
