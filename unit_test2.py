import unittest
from calculator import *

def canonical_test(strings):
    cano = parser(tokenizer(strings)).canonical()
    print('Canonical form = ', cano)
    print()
    
def diff_test(strings):
    fx = parser(tokenizer(strings)).canonical().diff('x').canonical()
    fy = parser(tokenizer(strings)).canonical().diff('y').canonical()
    print('Derivative form(x) = ', fx)
    print('Derivative form(y) = ', fy)
    print()

def eval_test(strings, s1, s2):
    x = parser(tokenizer(s1)).canonical().value
    y = parser(tokenizer(s2)).canonical().value
    value = parser(tokenizer(strings)).canonical().evaluate(x,y)
    print('Evaluation value = ', value)
    print()
    
def plot_test(strings):
    x_range = np.linspace(-10,10,399)
    res = plot_eval(strings, x_range)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(res[0], res[1])
    plt.subplot(1,2,2)
    plt.plot(res[2], res[3])
    plt.show()
    
    
    
class UnitTest(unittest.TestCase):
    def test1(self):
        s='x/((x)^2)^0.5+((x-0.100014)/((x-0.100011)^2)^0.5)*0.100067'
        s1='1'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = 1.10007')
        plot_test(s)
        print('\n\n')

    def test2(self):
        s='x^2*e^(-x)+pi^x'
        s1='2'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = 10.4109')
        plot_test(s)
        print('\n\n')

    def test3(self):
        s='sin(log(x^2-1))'
        s1='2'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = 0.89058')
        plot_test(s)
        print('\n\n')

    def test4(self):
        s='x^(x^2+1/x)'
        s1='2'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = 22.62742')
        plot_test(s)
        print('\n\n')

    def test5(self):
        s='1/(sin(x)-tan(x))'
        s1='1'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = -1.39677')
        plot_test(s)
        print('\n\n')

    def test6(self):
        s='x*sin(1/x)'
        s1='2'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = 0.95885')
        plot_test(s)
        print('\n\n')

    def test7(self):
        s='x^2-x*x+e^(log(x))'
        s1='1'
        s2='0'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = 1')
        plot_test(s)
        print('\n\n')

    def test8(self):
        s='x^2+y^2-9'
        s1='1'
        s2='2'
        print('input = ', s)
        canonical_test(s)
        diff_test(s)
        eval_test(s, s1, s2)
        print('Answer = -4')
        plot_test(s)
        print('\n\n')

    
if __name__ == "__main__":
    #unittest.main(argv=['first-arg-is-ignored'], exit=False) # for jupyter
    unittest.main()