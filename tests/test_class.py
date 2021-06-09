# content of test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "split")

    def test_three(self):
        assert 1
    
    def test_four(self):
        assert 1

