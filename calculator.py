import math as m
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr


class Node:
    def __init__(self):
        self.subset = []    
    def push(self, add):
        if type(add) == list:
            for i in add:
                self.subset.append(i)
        else:
            self.subset.append(add)
            
class AddNode(Node):
    def __str__(self):
        output = ''
        if len(self.subset) > 1:
            output += '('+str(self.subset[0])+')'
            for i in self.subset[1:]:
                output += '+'
                output += '('+str(i)+')'
            return output
        else:
            return str(self.subset[0])
    
    def canonical(self):
        # 일단 add_node에 모두 추가 -> [(Num), (AddNode), (var)]
        add_node = AddNode()
        for subset in self.subset:
            add_node.push(subset.canonical())                         # canonical 가능한지점까지 한다음 push

        # 추가된 것들 중 다시 더할 것 있으면 처리 -> [(2), (x+1), (y)] -> [(2), (y), (x), (1)]
        for i in range(len(add_node.subset)):
            if isinstance(add_node.subset[i], AddNode):               # 각 term 에 또 더할것이 있으면               
                temp = add_node.subset.pop(i).subset                 # 타입 변경해서 push 해주려고 .subset 함
                add_node.push(temp)
                return add_node.canonical()
        
        ## (add_node.subset 중에 더할것 없는 상태임)
        # add_node 의 항들을 같은 변수끼리 묶어 합치기(canonical)
        tmp_dic = {'&': 0}        
        for i in add_node.subset:                                     # add_node = [2*x, log(x)]
            if isinstance(i, NumNode):
                tmp_dic['&'] += i.value

            else:                                                      # mul, tril, log, var, power
                if isinstance(i, MulNode) and isinstance(i.subset[0], NumNode):             # 2*x, 3*log(x), ...
                    coeff = i.subset.pop(0)                            # 2
                    term_node = i                                       # x
                    
                    # tmp_dic 에 현재 항과 동일한 변수가 있는지 확인
                    if str(term_node) not in tmp_dic.keys():
                        tmp_dic[str(term_node)] = (term_node, coeff)    # tmp_dic ={'x':(x, 2)} # 없으면 새로 추가
                    else:
                        add_coef = AddNode()
                        add_coef.push(tmp_dic[str(term_node)][1])        # 있으면 기존에 있는 같은 변수(x)의 coeff 를 add_node에 push
                        add_coef.push(coeff)                             # 현재의 coeff 도 push
                        add_coef = add_coef.canonical() 
                        tmp_dic[str(term_node)] = (term_node, add_coef)  # tmp_dic = {'x':(x, 5)}
                else:                                                   # log(x)
                    # tmp_dic에 현재 항과 동일한 변수가 있는지 확인
                    if str(i) not in tmp_dic.keys():
                        tmp_dic[str(i)] = (i, NumNode(1))                # tmp_dic['log(x)'] = (log(x), 1)
                    else:
                        add_coef = AddNode()
                        add_coef.push(tmp_dic[str(i)][1])                 # 기존에 있는 key ('log(x)')의 coeff (ex.2) 를 add
                        add_coef.push(NumNode(1))                         # 현재 나의 coeff (1) add
                        add_coef = add_coef.canonical()      
                        tmp_dic[str(i)] = (i, add_coef)                   # 기존에 있던 key 는 새로운 value 가짐 # tmp_dic['log(x)'] = (log(x), 3)

        sorting = list(tmp_dic.keys())
        sorting.sort()

        ## (tmp_dic 생성 이후 후처리)
        # 계수가 0인 항 삭제
        for i in sorting[1:]:
            if isinstance(tmp_dic[i][1], NumNode):
                if tmp_dic[i][1].value == 0:
                    tmp_dic.pop(i)

        # 상수항만 있으면 상수항 리턴
        if len(tmp_dic) == 1:
            return NumNode(tmp_dic['&'])
        
        # 상수항 0이면 상수항 제거
        elif tmp_dic['&'] == 0: 
                tmp_dic.pop('&')

        ## (상수항 아닌 남은 항들 canonical)
        # 하나만 남았을때는 계수*변수 곱해서 mul_node 리턴
        if len(tmp_dic) == 1:                           # tmp_dic = {'x':(x, 3)}
            #i = tmp_dic.keys()[0]
            i = list(tmp_dic.keys())[0]                 # 'x'
            mul_node = MulNode()
            mul_node.push(tmp_dic[i][0])                # mulnode ('x')
            mul_node.push(tmp_dic[i][1])                # 3
            return mul_node.canonical()                # 3*x
        # 하나이상 남았을 때는 계수*변수 곱하여 push, add_node 리턴
        else:                                          # tmp_dic = {'x':(mulnode, 3), 'x^2':(mulnode, 5)}
            add_node.subset = []
            sorting = list(tmp_dic.keys())
            sorting.sort()
            for i in sorting:                          # ['&','x','x^2']
                if i == '&':                           # 상수항일때
                    add_node.push(NumNode(tmp_dic[i]))
                else:                                  # 상수항 아닐때
                    mul_node = MulNode()
                    mul_node.push(tmp_dic[i][1])        # 5
                    mul_node.push(tmp_dic[i][0])        # x^2
                    add_node.push(mul_node.canonical()) # 5*x^2

            return add_node                            # [3*x, 5*x^2]

    def diff(self, var):
        temp = []
        for i in range(len(self.subset)):              # 각 항을 미분한 다음 더해
            temp.append(self.subset[i].diff(var))
        add_node = AddNode()
        add_node.push(temp)
        return add_node

    def evaluate(self, x, y):                            # 각 항에 x or y 값 넣은 다음 더해
        add_node = AddNode()
        for i in range(len(self.subset)):
            add_node.push(self.subset[i].evaluate(x,y))
        return add_node.canonical()



class MulNode(Node):
    def __str__(self):
        output = ''
        if len(self.subset) > 1:
            output += '('+str(self.subset[0])+')'
            for i in self.subset[1:]:
                output += '*'
                output += '('+str(i)+')'
            return output
        else:
            return str(self.subset[0])
        
    def canonical(self):
        #일단 mul_node 에 모두 추가
        mul_node = MulNode()
        for subset in self.subset:
            mul_node.push(subset.canonical())  
            
        # 추가된 것들 중 다시 곱할 것 있으면 처리 
        for i in range(len(mul_node.subset)):                
            if isinstance(mul_node.subset[i], MulNode):       # (2*x)*x^2 -> [2*x, x^2]
                mul_subset = mul_node.subset.pop(i).subset  # mul_subset = 2*x
                mul_node.push(mul_subset)                     # result = [x^2, 2, x]
                return mul_node.canonical()
        
        # 분배법칙
        for i in range(len(mul_node.subset)):                
            if isinstance(mul_node.subset[i], AddNode):       # [(x+1), x^2]  의 (x+1)
                add_node = AddNode()
                add_subset = mul_node.subset.pop(i).subset  # add_subset = (x+1)
                for j in add_subset:                         # mul_node [(x+1), x^2] -> add_node [x*x^2, 1*x^2]
                    temp = MulNode()
                    temp.push(j)                               
                    temp.push(mul_node.subset)#[i])           
                    add_node.push(temp)                        
                return add_node.canonical()
        
        
        ## (mul_node.subset 내에 (a+b)*x 또는 (a*b)*x 없는상태)
        # mul_node 의 항들을 같은 변수끼리 묶어 합치기(canonical)
        tmp_dic = {'&': 1}                 
        for i in mul_node.subset:                            
            if isinstance(i, PowerNode):                      # x^2 * x^3 -> x^(2+3)    # powernode는 base, expo로 접근해야해서 if문 따로
                if str(i.base) not in tmp_dic.keys():        # key 에 base 없으면 추가
                    tmp_dic[str(i.base)] = (i.base, i.exponent)
                else:                                         # key 에 base 있으면
                    add_exp = AddNode()
                    add_exp.push(tmp_dic[str(i.base)][1])      # 기존 항의 지수 +
                    add_exp.push(i.exponent)                   # 현재 항의 지수
                    add_exp = add_exp.canonical()
                    tmp_dic[str(i.base)] = (i.base, add_exp)   # tmp_dic['x'] = {x, 5}

            elif isinstance(i, Term):                          # tri, log, inverse, var  # ex. sin(x)  #여기서는 Term 이 mulnode 인 경우가 없음, 위에서 다 처리해줘서
                if str(i) not in tmp_dic.keys():
                    tmp_dic[str(i)] = (i, NumNode(1))          # tmp_dic = {'sin(x)': sin(x), 1}
                else:
                    add_exp = AddNode()
                    add_exp.push(tmp_dic[str(i)][1])           # 기존 항의 지수 +
                    add_exp.push(NumNode(1))                   # 현재 항의 지수
                    add_exp = add_exp.canonical()              
                    tmp_dic[str(i)] = (i, add_exp)             # tmp_dic['sin(x)'] = (sin(x), 2)
            else:                                              
                tmp_dic['&'] *= i.value

        sorting = list(tmp_dic.keys())
        sorting.sort()
        
        ## (tmp_dic 생성 이후 후처리)
        sorting.pop(0)                            ##상수항 제외하고 for문 처리하려고
        for i in sorting:                        # 계수가 0인 항 있으면 삭제 (x^0) - 곱해봤자 1
            if isinstance(tmp_dic[i][1], NumNode):
                if tmp_dic[i][1].value == 0:
                    tmp_dic.pop(i)

        if len(tmp_dic) == 1:                     # 항이 1개면 숫자 리턴
            return NumNode(tmp_dic['&'])
        elif tmp_dic['&'] == 0:                   # 상수항 0이면 0 리턴 - 곱해봤자 0
            return NumNode(0)
        elif tmp_dic['&'] == 1:                   # 상수항 1이면 상수항 제거 - 곱해봤자 1
            tmp_dic.pop('&')

        
        # 상수항 처리하고 나서 1개만 남을 경우
        if len(tmp_dic) == 1:                                                          # {'x':(x, 3)},  
            i = list(tmp_dic.keys())[0]
            return PowerNode(tmp_dic[i][0], tmp_dic[i][1]).canonical()                # {'x': (x,3)} 인 경우 x^3 
        # 상수항 처리하고 나서 항 여러개 남은 경우
        else:                                       
            mul_node.subset = []
            sorting = list(tmp_dic.keys())
            sorting.sort()
            for i in sorting:
                if i == '&':
                    mul_node.push(NumNode(tmp_dic[i]))
                else:
                    mul_node.push(PowerNode(tmp_dic[i][0], tmp_dic[i][1]).canonical()) # {'x':(x,3), 'log(x)':(log(x),2)} -> [x^3 * (log*(x))^2] 
            return mul_node

    def diff(self, var):
        add_node = AddNode()
        for i in range(len(self.subset)):          # x*log(x)*sin(x) -> log(x)*sin(x) + x*(1/x)*sinx + x*log(x)*cos(x)
            mul_node = MulNode()
            for j in range(len(self.subset)):
                if j == i:
                    mul_node.push(self.subset[j].diff(var)) # 미분할 항
                else:
                    mul_node.push(self.subset[j])
            add_node.push(mul_node)
        return add_node

    def evaluate(self, x, y):                        # 각 항에 x or y 값 넣은 다음 곱해
        mul_node = MulNode()
        for i in range(len(self.subset)):
            mul_node.push(self.subset[i].evaluate(x,y))
        return mul_node.canonical()


            

class NonNode:
    def __init__(self, term):
        self.term = term


class NegativeNode(NonNode):
    def __str__(self):
        return '-({})'.format(self.term)                                               
    def canonical(self):
        mul_node = MulNode()
        mul_node.push(NumNode(-1))
        mul_node.push(self.term)
        return mul_node.canonical()
    def diff(self, var):
        return NegativeNode(self.term.diff(var))
    def evaluate(self, x, y):
        return NegativeNode(self.term.evaluate(x,y)).canonical()



class Term:
    pass

class InverseNode(Term, NonNode):
    def __str__(self):
        return '(1)/({})'.format(self.term)
    def canonical(self):
        _denominator = self.term.canonical()
        if isinstance(_denominator, NumNode):          # 분모가 NumNode 일때와 아닐때 
            if abs(1/float(_denominator.value)) > 10**10:
                raise ValueError('Infinity')
            else:
                return NumNode(1/float(_denominator.value))
        else:
            return InverseNode(_denominator)
    def diff(self, var):                               # (1/f(x))' -> -(f(x)'/(f(x)^2))
        denom = InverseNode(PowerNode(self.term, NumNode(2)))
        mul_node = MulNode()
        mul_node.push(NumNode(-1))
        mul_node.push(denom)
        mul_node.push(self.term.diff(var))
        return mul_node
    
    def evaluate(self, x, y):
        return InverseNode(self.term.evaluate(x,y)).canonical()


class SinNode(Term, NonNode):
    def __str__(self):
        return 'sin({})'.format(self.term)
    def canonical(self):
        _sin_value = self.term.canonical()
        if isinstance(_sin_value, NumNode):              # sin(x) 의 (x)가 num or not
            if abs(m.sin(_sin_value.value)) > 10**10:
                raise ValueError('Infinity')
            else:
                return NumNode(m.sin(_sin_value.value))
        else:
            return  SinNode(_sin_value)
    def diff(self, var):
        mul_node = MulNode()
        mul_node.push(CosNode(self.term))
        mul_node.push(self.term.diff(var))
        return mul_node
    def evaluate(self, x, y):
        return SinNode(self.term.evaluate(x,y)).canonical()

class CosNode(Term, NonNode):
    def __str__(self):
        return 'cos({})'.format(self.term)
    def canonical(self):
        _cos_value = self.term.canonical()
        if isinstance(_cos_value, NumNode):               # cos(x) 의 (x)가 num or not
            if abs(m.cos(_cos_value.value)) > 10**10:
                raise ValueError('Infinity')
            else:
                return NumNode(m.cos(_cos_value.value))
        else:
            return CosNode(_cos_value)
    def diff(self, var):
        mul_node = MulNode()
        mul_node.push(NumNode(-1))
        mul_node.push(SinNode(self.term))
        mul_node.push(self.term.diff(var))
        return mul_node
    def evaluate(self, x, y):
        return CosNode(self.term.evaluate(x,y)).canonical()

class TanNode(Term, NonNode):
    def __str__(self):
        return 'tan({})'.format(self.term)
    def canonical(self):
        _tan_value = self.term.canonical()
        if isinstance(_tan_value, NumNode):                 # tan(x) 의 (x)가 num or not
            if abs(m.tan(_tan_value.value)) > 10**10:
                raise ValueError('Infinity')
            else:
                return NumNode(m.tan(_tan_value.value))
        else:
            return TanNode(_tan_value)
    def diff(self, var):
        mul_node = MulNode()        
        mul_node.push(PowerNode(CosNode(self.term), NumNode(-2)))
        mul_node.push(self.term.diff(var))
        return mul_node
    def evaluate(self, x, y):
        return TanNode(self.term.evaluate(x,y)).canonical()



class LogNode(Term, NonNode):
    def __str__(self):
        return 'log({})'.format(self.term)        
    def canonical(self):
        _log_value = self.term.canonical()
        if isinstance(_log_value, NumNode):                   # log(x) 의 (x)가 num or not
            if abs(m.log(_log_value.value)) > 10**10:
                raise ValueError('Infinity')
            else:
                return NumNode(m.log(_log_value.value))
        else:
            return LogNode(_log_value)
    def diff(self, var):                                      # log(f(x)) -> f'(x)/f(x)
        mul_node = MulNode()
        mul_node.push(InverseNode(self.term))
        mul_node.push(self.term.diff(var))
        return mul_node
    def evaluate(self, x, y):
        return LogNode(self.term.evaluate(x,y)).canonical()


class PowerNode:
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def __str__(self):
        return '({})^({})'.format(self.base, self.exponent)
    def canonical(self):
        # base, exp canonical 하고 시작
        _base = self.base.canonical()
        _exp = self.exponent.canonical()
    
        # 상수 되는것들 먼저 처리
        if isinstance(_base, NumNode) and isinstance(_exp, NumNode): # 2^2 
            try:
                temp = float(_base.value**_exp.value)
                return NumNode(temp)
            except:
                raise ValueError('Invalid number for power')
        
        # base 가 NumNode 일때 0, 1 되는 경우 처리
        if isinstance(_base, NumNode):                    # 0^(x), 1^(x), (-2)^(x) ...
            if _base.value == 0:
                return NumNode(0)
            elif _base.value == 1:
                return NumNode(1)
            elif _base.value < 0:
                return ValueError('Invalid number for power')

        # expo 가 NumNode 일때 1 or base or inverse 되는 경우 처리
        if isinstance(_exp, NumNode):                     # x^0, x^1, x^(-2) ...
            if _exp.value == 0:
                return NumNode(1)
            elif _exp.value == 1:
                return _base
            elif _exp.value < 0:                          # x^(-2) -> 1/(x^2)
                return InverseNode(PowerNode(_base, NumNode(_exp.value*(-1)))).canonical() # *(-1 ) : 다시 양수로 만들어주고 역수만들려고

        # base 가 powerNode 인 경우 지수끼리 곱한 후 재귀
        if isinstance(_base, PowerNode):                  # (2^x)^3  -> 2^(x*3)            
            mul_node = MulNode()
            mul_node.push(_base.exponent)       
            mul_node.push(_exp)                 
            return PowerNode(_base.base, mul_node).canonical()
        
    
        ## (base 에 powerNode 가 없는상태)
        # base 가 inverse 일때 처리
        if isinstance(_base, InverseNode):                # (1/x)^2 -->  1/x^2 
            return InverseNode(PowerNode(_base.term, _exp)).canonical()

        # base 가 mulNode 일때 처리
        if isinstance(_base, MulNode):                    # (x*2)^3 -> x^3*2^3 
            mul_node = MulNode()
            for subset in _base.subset:
                mul_node.push(PowerNode(subset, _exp)) 
            return mul_node.canonical()

        # exp 가 NumNode
        if isinstance(_exp, NumNode):     
            # exp 가 int 이고 base 가 addnode 일때 -> 지수, 밑 전개
            if (_exp.value > 1) and (_exp.value == int(_exp.value)): # for 문 돌면서 곱하려면 exp = int 여야함
                if isinstance(_base, AddNode):                        # (x+1)^3 -> (x+1)*(x+1)*(x+1)
                    mul_node = MulNode()
                    for i in range(int(_exp.value)):
                        mul_node.push(_base)
                    return mul_node.canonical()        

            return ExpNumNode(_base, _exp)                  # x^0.5, x^2, log(x)^2 ...
        # exp 에 변수 존재
        else:
            return ExpVarNode(_base, _exp)                  # x^x, 2^x, (x+1)^x ...
        
        
    def evaluate(self, x, y):
        _base = self.base.evaluate(x,y)
        _exp = self.exponent.evaluate(x,y)
        return PowerNode(_base, _exp).canonical()

    def diff(self, var):
        return self.canonical().diff(var)


class ExpNumNode(PowerNode):   
    def diff(self, var):                       # (2*x)^0.5 -> 0.5 * 2 *(2*x)^(-0.5)
        _exp = NumNode(self.exponent.value-1)  
        mul_node = MulNode()
        mul_node.push(self.exponent)           
        mul_node.push(PowerNode(self.base, _exp))
        mul_node.push(self.base.diff(var))                    
        return mul_node

class ExpVarNode(PowerNode):
    def diff(self, var):                        # (x+1)^(2*x)
        mul_1 = MulNode()                       # diff 전에 a^b -> e^(log(a)*b) 꼴로 바꿔줌
        mul_1.push(self.exponent)
        mul_1.push(LogNode(self.base))
        self.base = NumNode(e)
        self.exponent = mul_1.canonical()
        
        mul_node = MulNode()                    
        mul_node.push(self)                     
        mul_node.push(self.exponent.diff(var))   #  e^(log(a)*b) --> e^(log(a)*b) * (log(a)*b)'
        return mul_node
        
    
class VarNode:
    def __init__(self):
        pass
    def canonical(self):
        return self
    def diff(self, var):     
        if str(self) == var:
            return NumNode(1)
        else:
            return NumNode(0)

class VarNode_x(Term, VarNode):
    def __str__(self):
        return 'x'
    def evaluate(self, x, y):
        return NumNode(x)

class VarNode_y(Term, VarNode):
    def __str__(self):
        return 'y'
    def evaluate(self, x, y):
        return NumNode(y)

class NumNode:
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return '{}'.format(float(self.value))
    def canonical(self):
        if abs(self.value) > 10**10:
            raise ValueError('Infinity')
        else:
            return self
    def evaluate(self, x, y):
        return self
    def diff(self, var):
        return NumNode(0)
        
# =========================================================================== 
# Tokenizing & Parsing
# =========================================================================== 
        
e = m.e
pi = m.pi

func_list = ['sin','cos','tan','log']
const_list = ['e','pi']
var_list = ['x','y']

def tokenizer(A):

    A = A.strip()
    if len(A) == 0: return 'empty input'

    single = ['+','-','*','/','(',')','^','x','y','e']
    num = ['0','1','2','3','4','5','6','7','8','9']
    word_2 = ['pi']
    word_3 = ['sin','cos','tan','log']

    tokens = []
    loc = 0
    
    len_A = len(A)

    while(loc < len_A):
        if A[0:3] in word_3:
            tokens.append(A[0:3])
            loc += 3
            A = A[3:]
        elif A[0:2] in word_2:
            tokens.append(A[0:2])
            loc += 2
            A = A[2:]
        elif A[0] in single:
            tokens.append(A[0])
            loc += 1
            A = A[1:]
        elif A[0] in num:                 
            temp = 1
            while(A[temp:temp+1] in num):
                temp += 1
            if A[temp:temp+1] == '.':
                temp += 1
                while (A[temp:temp+1] in num):
                    temp += 1
            tokens.append(A[0:temp])
            loc += temp
            A = A[temp:]
        elif A[0] == ' ':
            loc += 1
            A = A[1:]

    tokens.append('$')
    return tokens


def parser(tokens):
    node = takeAddNode(tokens)
    takeIt(tokens, ['$'])
    return node

def takeAddNode(tokens):                      # AddNode -> AddNode | MulNode (+- MulNode)
    mul_node = takeMulNode(tokens)
    if tokens[0] in ['+','-']:                # '+' 다 소화될때까지 add_node 리턴
        add_node = AddNode()
        add_node.push(mul_node)
        while(tokens[0] in ['+','-']):
            op = takeIt(tokens, ['+','-'])
            mul_node = takeMulNode(tokens)
            if op == '+':
                add_node.push(mul_node)
            else:
                add_node.push(NegativeNode(mul_node))
        return add_node
    else:
        return mul_node

def takeMulNode(tokens):                      # MulNode -> MulNode | FuncNode (*/ FuncNode)
    func_node = takeFuncNode(tokens)
    if tokens[0] in ['*','/']:                # '*' 다 소화될때까지 mul_node 리턴
        mul_node = MulNode()
        mul_node.push(func_node)
        while(tokens[0] in ['*','/']):
            op = takeIt(tokens, ['*','/'])
            func_node = takeFuncNode(tokens)
            if op == '*':
                mul_node.push(func_node)
            else:
                mul_node.push(InverseNode(func_node))
        return mul_node
    else:
        return func_node



def takeFuncNode(tokens): # FuncNode -> -FuncNode | 'sin' FuncNode | 'cos' FuncNode | 'tan' FuncNode | 'log' FuncNode | TerminalNode
    if tokens[0] == '-':
        takeIt(tokens, ['-'])
        return NegativeNode(takeFuncNode(tokens))
    elif tokens[0] in func_list:
        func = takeIt(tokens, func_list)
        if func == 'sin':
            return SinNode(takeFuncNode(tokens))
        elif func == 'cos':
            return CosNode(takeFuncNode(tokens))
        elif func == 'tan':
            return TanNode(takeFuncNode(tokens))
        elif func == 'log':
            return LogNode(takeFuncNode(tokens))
    else:
        return takeTerminalNode(tokens)                                           

const_dic = {'e':e, 'pi':pi}

def takeTerminalNode(tokens):                      # TerminalNode -> (AddNode)(^FuncNode) | Var(^FuncNode) | Num(^FuncNode) | const(^FuncNode)
    if tokens[0] == '(':          # (AddNode)
        takeIt(tokens, ['('])
        base = takeAddNode(tokens)                                         
        takeIt(tokens, [')'])
    elif tokens[0] in var_list:   # var
        var = takeIt(tokens, var_list)
        if var == 'x':
            base = VarNode_x()
        else:
            base = VarNode_y()
    elif tokens[0] in const_list: # e, pi
        base = NumNode(const_dic[takeIt(tokens, const_list)])
    else:                         # num
        base = NumNode(takeIt(tokens, 'Num'))

    if tokens[0] == '^':          # (^FuncNode)
        takeIt(tokens, ['^'])
        exponent = takeFuncNode(tokens)
        return PowerNode(base, exponent)
    else:
        return base

def takeIt(tokens, token_type):
    temp = tokens.pop(0)
    if token_type == 'Num':
        return float(temp)
    else:
        if temp in token_type:
            return temp
        else:
            raise ValueError('Invalid token type')

# =========================================================================== 
# Execute & Plot
# =========================================================================== 

import copy
def execute(strings, s1, s2):
    try: 
        tokens = tokenizer(strings)
    except:
        return ('Invalid Letters','nan','nan','nan','nan','nan','nan')

    try: # parsed node
        expr = parser(tokens)
        expr_cp = copy.deepcopy(expr) # 이후 canonical, diff 등으로 값 바뀌니까 복사한 값 return 함
    except:
        return ('Invalid Equation','nan','nan','nan','nan','nan','nan')
    
    try: # canonicalized node
        cano_expr = expr.canonical()
        cano_expr_cp = copy.deepcopy(cano_expr)
    except:
        return (str(expr_cp), 'CanonicalError','nan','nan','nan','nan','nan')

    try: # diff node
        fx = cano_expr.diff('x').canonical()
        fy = cano_expr.diff('y').canonical()
        fx_cp = copy.deepcopy(fx)
        fy_cp = copy.deepcopy(fy)
    except:
        return (str(expr_cp), str(cano_expr_cp), 'DiffError','nan','nan','nan','nan')

    try: # x,y 값을 node로 바꿈
        x = parser(tokenizer(s1)).canonical().value
        y = parser(tokenizer(s2)).canonical().value
    except:
        return (str(expr_cp), str(cano_expr_cp), str(fx_cp), str(fy_cp), 'Invaild Value(x or y)','nan','nan')

    try: # x,y 값 넣어서 계산
        func_val = cano_expr.evaluate(x,y)#.canonical()
        func_val_cp = copy.deepcopy(func_val)
    except:
        return (str(expr_cp), str(cano_expr_cp), str(fx_cp), str(fy_cp), 'Discontinuity','nan','nan')

    try: # 대입한 x, y 에서 미분가능성 확인
        temp = fx.evaluate(x,y)#.canonical()
        temp = fy.evaluate(x,y)#.canonical()
    except:
        return (str(expr_cp), str(cano_expr_cp), str(fx_cp), str(fy_cp), str(func_val_cp),'Continuity', 'Non-differentiable')

    return (str(expr_cp), str(cano_expr_cp), str(fx_cp), str(fy_cp), str(func_val_cp), 'Continuity','Differentiable')


def plot_eval(strings, xrange):
    x1,y1,x2,y2 = [],[],[],[]
    fx = parser(tokenizer(strings)).canonical().diff('x').canonical()
    fy = parser(tokenizer(strings)).canonical().diff('y').canonical()

    if (str(fy)) == ('0.0'):
        for i in xrange:
            try:
                j = parser(tokenizer(strings)).canonical().evaluate(i,0).canonical().value
                j = float(j)
                if m.isnan(j):
                    raise ValueError('plot_eval_Error')
                x1.append(i)
                y1.append(j)
            except:
                continue
        for i in xrange:
            try:
                j = fx.evaluate(i,0).canonical().value
                j = float(j)
                if m.isnan(j):
                    raise ValueError('plot_eval_Error')
                x2.append(i)
                y2.append(j)
            except:
                continue
        return (x1, y1, x2, y2)        
    else:
        return ([-100],[-100],[-100],[-100])

    
def main(Equation, x_val, y_val):
    res = execute(Equation, x_val, y_val)
    res_plot = plot_eval(Equation, np.linspace(-10, 10, 400)) 
    plt.clf()
    plt.figure(figsize=(8,10))
    plt.subplot(2,1,1)
    plt.plot(res_plot[0], res_plot[1])
    plt.subplot(2,1,2)
    plt.plot(res_plot[2], res_plot[3])
    return (res[0],res[1],res[2],res[3],res[4],res[5],res[6],plt)
    
    
    
if __name__ == "__main__":
    demo = gr.Interface(fn=main, inputs=["text", "text", "text"], \
                    outputs=[gr.Textbox(label="Input"),\
                             gr.Textbox(label="Canonicalize"),\
                             gr.Textbox(label="Diff_x"),\
                             gr.Textbox(label="Diff_y"),\
                             gr.Textbox(label="Evaluation"),\
                             gr.Textbox(label="Continuity"),\
                             gr.Textbox(label="Differentiable"),\
                             gr.Plot(label="Plot")])
                             
    demo.launch(server_name="0.0.0.0")