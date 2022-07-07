import unittest
from calculator import *

def canonical_test(strings):
    cano = parser(tokenizer(strings)).canonical()
    return cano
    
def diff_test(strings):
    fx = parser(tokenizer(strings)).canonical().diff('x').canonical()
    fy = parser(tokenizer(strings)).canonical().diff('y').canonical()
    return [fx,fy]


def eval_test(strings, s1, s2):
    x = parser(tokenizer(s1)).canonical().value
    y = parser(tokenizer(s2)).canonical().value
    value = parser(tokenizer(strings)).canonical().evaluate(x,y)
    return value

def get_anslist(location):
    with open(location, 'r') as f:
        answers = f.readlines()

    ans, tmp = [], []
    for i in range(len(answers)):
        if answers[i]=='\n':
            ans.append(tmp)
            tmp=[]
            continue
        else:
            tmp.append(answers[i][:-1])
    return ans

    
class UnitTest(unittest.TestCase):
    def test1(self):
        s='x/((x)^2)^0.5+((x-0.100014)/((x-0.100011)^2)^0.5)*0.100067'
        s1='1'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)

        self.assertEqual(str(cano), ans[0][0])
        self.assertEqual(str(fx), ans[0][1])
        self.assertEqual(str(fy), ans[0][2])
        self.assertEqual(str(value), ans[0][3])

    def test2(self):
        s='x^2*e^(-x)+pi^x'
        s1='2'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[1][0])
        self.assertEqual(str(fx), ans[1][1])
        self.assertEqual(str(fy), ans[1][2])
        self.assertEqual(str(value), ans[1][3])

    def test3(self):
        s='sin(log(x^2-1))'
        s1='2'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[2][0])
        self.assertEqual(str(fx), ans[2][1])
        self.assertEqual(str(fy), ans[2][2])
        self.assertEqual(str(value), ans[2][3])

    def test4(self):
        s='x^(x^2+1/x)'
        s1='2'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[3][0])
        self.assertEqual(str(fx), ans[3][1])
        self.assertEqual(str(fy), ans[3][2])
        self.assertEqual(str(value), ans[3][3])

    def test5(self):
        s='1/(sin(x)-tan(x))'
        s1='1'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[4][0])
        self.assertEqual(str(fx), ans[4][1])
        self.assertEqual(str(fy), ans[4][2])
        self.assertEqual(str(value), ans[4][3])

    def test6(self):
        s='x*sin(1/x)'
        s1='2'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[5][0])
        self.assertEqual(str(fx), ans[5][1])
        self.assertEqual(str(fy), ans[5][2])
        self.assertEqual(str(value), ans[5][3])

    def test7(self):
        s='x^2-x*x+e^(log(x))'
        s1='1'
        s2='0'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[6][0])
        self.assertEqual(str(fx), ans[6][1])
        self.assertEqual(str(fy), ans[6][2])
        self.assertEqual(str(value), ans[6][3])

    def test8(self):
        s='x^2+y^2-9'
        s1='1'
        s2='2'
        cano = canonical_test(s)
        [fx, fy] = diff_test(s)
        value = eval_test(s, s1, s2)
        
        self.assertEqual(str(cano), ans[7][0])
        self.assertEqual(str(fx), ans[7][1])
        self.assertEqual(str(fy), ans[7][2])
        self.assertEqual(str(value), ans[7][3])

    
if __name__ == "__main__":    
    ans = get_anslist('./unit_test_answer.txt')
    #unittest.main(argv=['first-arg-is-ignored'], exit=False)
    unittest.main()