from symbolica_community import Expression, S,E
from symbolica_community.tensors import TensorNetwork,Representation,TensorStructure,TensorIndices,Tensor
import symbolica_community
import symbolica_community.tensors as tensors
import random


mu = E("mink(4,mu)")
nu = E("mink(4,nu)")
i = E("bis(4,i)")
j = E("bis(4,j)")
k = E("bis(4,k)")
gamma,p,w,mq,id = S("γ","P","W","mq","id")
x = gamma(mu,i,k)*(p(2,nu)*gamma(nu,k,j)+mq*id(k,j))*w(1,i)*w(3,mu)
tn = TensorNetwork(x)

tn.contract()
t = tn.result()
print(t)

x = gamma(mu,i,k)
g = TensorNetwork(x).result()
print(g)
print(g.structure())
params = [Expression.I]
params += TensorNetwork(w(1,i)).result()
params += TensorNetwork(w(3,mu)).result()
params += TensorNetwork(p(2,nu)).result()
constants = {mq: E("173")}
e=t.evaluator(constants=constants, params=params, funs={})
c = e.compile(function_name="f", filename="test_expression.cpp",
              library_name="test_expression.so", inline_asm=False)


e_params = [random.random()+1j*random.random() for i in range(len(params))]

print(c.evaluate_complex([e_params])[0])

print(c.evaluate_complex([e_params])[0].structure())
tn = TensorNetwork(p(1,mu)*w(1,mu))
tn.contract()
t=tn.result()
t.to_dense()
print(t)
print(t.structure())



lor = Representation("lor",4,dual = True)
t = tensors.sparse_empty([lor,lor],type(gamma))
t[6]=E("f(x)*(1+y)")
t[[3,2]]=E("sin(alpha)")
t.to_dense()
print(t)
print(t.structure())

t = tensors.dense([3,3],[0,0,123,
                        11,3,234,
                        234,23,44,])
a=t[1:2]
t[[1,2]]=3/34
# t.to_dense()
print(t)
print(t.structure())

i = Representation("bis",4)
print(TensorStructure(mu,i,i,name=gamma))
g = tensors.dense(TensorStructure(lor,i,i,name=gamma),[0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,

                                                        0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,

                                                        0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,
                                                                                                                                                            0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,
                                                        0,0,0,0,      ]
)
print(g)
repr(g.structure())
