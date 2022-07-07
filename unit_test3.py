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


class UnitTest(unittest.TestCase):
    def unittest_cano(self):
        self.assertEqual(str(canonical_test('sin(log(x^2-1))')), 'sin(log((-1.0)+((x)^(2.0))))')
        self.assertEqual(str(canonical_test('x*sin(1/x)')), '(sin((1)/(x)))*(x)')
        self.assertEqual(str(canonical_test('x^2-x*x+e^(log(x))')), '(2.718281828459045)^(log(x))')
        self.assertEqual(str(canonical_test('x^2+y^2-9')), '(-9.0)+((x)^(2.0))+((y)^(2.0))')
        
    def unittest_diffx(self):
        self.assertEqual(str(diff_test('sin(log(x^2-1))')[0]), '(2.0)*((1)/((-1.0)+((x)^(2.0))))*(cos(log((-1.0)+((x)^(2.0)))))*(x)')
        self.assertEqual(str(diff_test('x*sin(1/x)')[0]), '((-1.0)*((1)/((x)^(2.0)))*(cos((1)/(x)))*(x))+(sin((1)/(x)))')
        self.assertEqual(str(diff_test('x^2-x*x+e^(log(x))')[0]), '((1)/(x))*((2.718281828459045)^(log(x)))')
        self.assertEqual(str(diff_test('x^2+y^2-9')[0]), '(2.0)*(x)')
        
    def unittest_diffy(self):
        self.assertEqual(str(diff_test('sin(log(x^2-1))')[1]), '0.0')
        self.assertEqual(str(diff_test('x*sin(1/x)')[1]), '0.0')
        self.assertEqual(str(diff_test('x^2-x*x+e^(log(x))')[1]), '0.0')
        self.assertEqual(str(diff_test('x^2+y^2-9')[1]), '(2.0)*(y)')
        
    def unittest_eval(self):
        self.assertEqual(str(eval_test('sin(log(x^2-1))',s1='2',s2='0')), '0.8905770416677471')
        self.assertEqual(str(eval_test('x*sin(1/x)',s1='2',s2='0')), '0.958851077208406')
        self.assertEqual(str(eval_test('x^2-x*x+e^(log(x))',s1='1',s2='0')), '1.0')
        self.assertEqual(str(eval_test('x^2+y^2-9'),s1='1',s2='2'), '-4.0')
        
        
        
if __name__ == "__main__":    
    #unittest.main(argv=['first-arg-is-ignored'], exit=False)
    unittest.main()
