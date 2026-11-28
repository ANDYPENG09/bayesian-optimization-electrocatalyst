import importlib.util
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location("rag",Path(__file__).resolve().parents[1]/"rag.py")
rag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag)

class TestRetrieval(unittest.TestCase):
    def test_citations_and_relevance(self):
        with tempfile.TemporaryDirectory() as folder:
            db=rag.database(Path(folder)/"test.sqlite")
            with db:
                rag.put(db,"paper1","Iridium oxide","10.0000/example",3,"pdf","IrOx oxygen evolution calcination temperature")
                rag.put(db,"paper2","Polymer membrane","",None,"abstract","PVA gas separation")
            hits=rag.search(db,"IrOx calcination temperature",1)
            self.assertEqual(hits[0]["source"],"paper1")
            self.assertEqual(hits[0]["page"],3)
            self.assertEqual(hits[0]["citation"],"E1")
            db.close()

    def test_empty(self):
        db=rag.database(":memory:")
        self.assertEqual(rag.search(db,"test"),[])
        self.assertEqual(rag.generate("test",[]),"No retrieved evidence; answer withheld.")
        db.close()

if __name__=="__main__":
    unittest.main()
