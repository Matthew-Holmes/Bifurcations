from math import factorial
from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import copy
import itertools

def latexify(x):
    out = '$' + x + '$'
    return out

def lprint(x):
    display(Markdown(latexify(latex(x))))

class ExpressionBlock:
    def __init__(self, c = 1, s_string = '', n = 0, p_dict = {}, xs_dict = {}, func1 = 'f', func2 = 'h', X = 'X', Y = 'Y'):
        self.c = c # constant
        self.s_string = s_string # partials, i.e. 'XXY'
        self.n = n # number of X inputs the expression has
        self.k = len(s_string) # even if no partials, still include an identity premultiple
        self.p_dict = p_dict # P functions, 0 --> id, otherwise is a partial of h
        self.xs_dict = xs_dict # which X's go into each P
        
        # the rest are just for the cosmetic output
        self.func1 = func1 
        self.func2 = func2
        self.X = X
        self.Y = Y
        
    def __str__(self):
        # overide representation so can be printed as latex
        if self.c == 0:
            return ""
        
        out = ""
        # add the constant
        if self.c != 1:
            out += str(self.c) + " "
            
        # add f
        out += self.func1
        
        # add partials if there are any
        if len(self.s_string) != 0:
            out += "_{"
            out += self.s_string
            out += "} "
            out += r"\cdot \left["
            

        for i in range(1,self.k + 1):
            if self.p_dict[i] == 0:
                out += "id("
            else:
                out += "h_{"
                out += "X"*len(self.xs_dict[i])
                out += "}("
                
            for xi in self.xs_dict[i]:
                out += "X_{" + str(xi) + "}, "
                
            # inner excess comma deletion
            out = out[:-2]
            
            out += "), "
        
        if len(self.s_string) != 0:
            out = out[:-2] # outer excess comma deletion
            out += r"\right]"       
            
                
        # and we are done, here just list out the h terms               
        return out

    def _latex_(self):
        # so works with lprint
        return str(self)
    
    def diff(self):
        # apply the rule we worked out before
        # returns a list of ExpressionBlock instances
        
        out = []
        # first half
        temp = copy.deepcopy(self)
        temp.n += 1
        temp.s_string += 'X'
        temp.k += 1
        temp.p_dict[temp.k] = 0 # identity
        temp.xs_dict[temp.k] = (temp.n,)
        out.append(temp)
        
        temp = copy.deepcopy(self)
        temp.n += 1
        temp.s_string += 'Y'
        temp.k += 1
        temp.p_dict[temp.k] = 1 # identity
        temp.xs_dict[temp.k] = (temp.n,)
        out.append(temp)
        
        # second half
        # loop over each term in the sum
        for i in range(1,self.k + 1):
            # ktuple = (gamma, k)
            # indexing starts from 0
            if self.p_dict[i] == 0:
                # differentiating id --> zeros out
                continue 
            
            temp = copy.deepcopy(self)
            temp.n += 1
            temp.p_dict[i] = temp.p_dict[i] + 1 # take a further derivative of h
            new_xs = list(temp.xs_dict[i])
            new_xs.append(temp.n)
            
            temp.xs_dict[i] = tuple(new_xs)

            out.append(temp)
            
        # and we are done
        return out
                        
           
class Expression:
    def __init__(self, blocks = [ExpressionBlock()]):
        if blocks is None:
            self.blocks = [] # python reasons - list is mutable
        else:
            self.blocks = blocks
            
    def __str__(self):
        if not self.blocks:
            # empty list
            return ""
        out = ""
        for block in self.blocks:
            # should be an instance of the ExpressionBlock class
            out += str(block) + " + "
        
        out = out[:-3] # remove last plus
        
        return out
    
    def _latex_(self):
        # so works with lprint
        return str(self)
    
    
    def diff(self):
        # returns a new Expression object that is the partial x derivative of the old one
        out = []
        for block in self.blocks:
            block_list = block.diff()
            
            # append to the list
            out += block_list
            
        return Expression(blocks = out)
                    
            
class SymbolicXYTensor():
        def __init__(self, x_dim, y_dim, xy_order):
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.xy_order = xy_order
            self.data = pd.DataFrame
            
            # the sets from which we draw the possible multi-indices
            x_dims = list(range(1, x_dim+1)) 
            y_dims = list(range(1, y_dim+1))
            x_dims = ['x' + str(dim) for dim in x_dims]
            y_dims = ['y' + str(dim) for dim in y_dims]

            iterables = [] 

            self.size = 1
            for space in self.xy_order:
                if (space == 'x') or (space == 'X'):
                    iterables.append(x_dims)
                    self.size = self.size*x_dim 
                elif (space == 'y') or (space == 'Y'):
                    iterables.append(y_dims)
                    self.size = self.size*y_dim
                else:
                    raise(Exception('invalid xy_order syntax'))

            multindex = pd.MultiIndex.from_product(iterables, names = list(range(1, len(xy_order) + 1)))

            self.data = pd.DataFrame(pd.Series(np.zeros(self.size), index = multindex), columns = ['data'] )
            
        def fill_from_function(self, function, var_dict, position):
            def row_func(row):
                partials = row.name # a tuple
                temp = function # will be differentiating
                #print(row.name)
                for partial in partials:
                    temp = temp.diff(var_dict[partial])
                    #lprint(temp)
                return temp(**position) # unpack tuple as coordinates
            
            def row_func_no_eval(row):
                partials = row.name # a tuple
                temp = function # will be differentiating
                #print(row.name)
                for partial in partials:
                    temp = temp.diff(var_dict[partial])
                    #lprint(temp)
                return temp # this is used when we know will be a constant
            
            if position is None:
                # bit hacky
                self.data['data'] = self.data.apply(row_func_no_eval, axis = 1)
            else:
                self.data['data'] = self.data.apply(row_func, axis = 1)
            
        def vec_mult(self, vec):
            if len(self.xy_order) == 1:
                # dual space case
                #print(self.data)
                return sum(self.data['data']*vec)
            
            out = SymbolicXYTensor(x_dim = self.x_dim, y_dim = self.y_dim, xy_order = self.xy_order[:-1])
            
            out.data = self.data.copy(deep = True)
            out.data['vec'] = list(vec)*int(out.size)
            out.data['data'] = out.data['data']*out.data['vec']
            out.data = out.data.drop(columns = 'vec')
            
            #print('preparing to gb')
            #print(out.data)
            
            out.data = out.data.groupby(level=[Integer(i) for i in range(Integer(1),Integer(len(self.xy_order)))]).sum()
            
            #print('after gb')
            #print(self.xy_order)
            #print(out.data)
            
            return out
            # for some reason only sage integers work here, who knows why
   
  
class SymbolicXYVectorTensor():
    # class that wraps up n tensors so they can represent multivariable domain functions
    # and evaluate to produce vectors as the lowest dimensional output
        def __init__(self, x_dim, y_dim, xy_order ='', vec_length = 1):
            
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.xy_order = xy_order
            self.vec_length = vec_length
            
            self.tensors = {}
            for i in range(0, vec_length):
                self.tensors[i] = SymbolicXYTensor(x_dim, y_dim, xy_order)
                
                
        def fill_from_functions(self, functions, var_dict, position):
            for i in range(0, self.vec_length):
                self.tensors[i].fill_from_function(functions[i], var_dict, position)
                
        def vec_mult(self, vec):
            #print(self.xy_order)
            if len(self.xy_order) == 1:
                out = [] # just in the output is vector case
                for i in range(0, self.vec_length):
                    out.append(self.tensors[i].vec_mult(vec))
                    
            else:
                # returning a lower dimensional tensor
                out = SymbolicXYVectorTensor(x_dim = self.x_dim, y_dim = self.y_dim,
                                             xy_order = self.xy_order[:-1], vec_length = self.vec_length)
                
                #print(self.tensors[0].data)
                #print(out.tensors[0].data)
                
                for i in range(0, self.vec_length):
                    out.tensors[i] = self.tensors[i].vec_mult(vec)
                    #print(out.tensors[i].data)
                    #print('vec multed')
                    
            return out
        
        def evaluate(self, vectors):
            # evaluate all the way to vector output
            if len(vectors) != len(self.xy_order):
                raise(Exception('wrong number of vector inputs provided!'))
                return
            
            current_tensors = self
            for vec in vectors[::-1]:
                current_tensors = current_tensors.vec_mult(vec)
                # collapse down the SymbolicXYVectorTensor objects
                
            if len(current_tensors) != self.vec_length:
                raise(Exception('something has gone wrong!'))
                
            return current_tensors # should just be a vector by this point
                
                
class TensorDict(dict):
    # upgraded dictionary class that fills in missing elements using the rules
        def __init__(self, funcs, position, var_dict, x_dim = 2, y_dim = 2, func1 = 'f', func2 = 'h', X = 'X', Y = 'Y'):
            super(TensorDict, self).__init__() # initialise empty dictionary
            
            self.funcs = funcs # list of functions that we will evaluate to get the tensors
            self.position = position # where eveything is evalated (bifurcation point?)
            self.var_dict = var_dict # the sage variables we'll be using
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.func1 = func1
            self.func2 = func2
            self.X = X
            self.Y = Y
            
        def __getitem__(self, key):
            # lookup a tensor, if its not there then build it
            try:
                return dict.__getitem__(self, key)
            except: # keyerror
                print('generating ' + key)
                tensor = SymbolicXYVectorTensor(x_dim = self.x_dim , y_dim = self.y_dim, xy_order = key[1:], vec_length = self.y_dim)
                # ignore the f/h at the start
                # this is the case for both derivatives of f,h : _ --> Y
                
                if key[0] == self.func1:
                    # just getting values of f_....
                    tensor.fill_from_functions(self.funcs, self.var_dict, self.position)
                elif key[0] == self.func2:
                    # this is h_kX derivative, not so easy to compute
                    raise(Exception(key + ' not possible, need to add the tensor dict manually'))
                    return 
                
                dict.__setitem__(self, key, tensor) # now save so we can reuse
                
                #print(tensor.tensors[0].data)
                return tensor
                
            
def get_fy_inv(x_dim, y_dim, funcs, var_dict, position):
    # first use the symbolic tensor class to compute the derivative
    A = SymbolicXYVectorTensor(x_dim, y_dim, xy_order = 'Y', vec_length = y_dim)
    A.fill_from_functions(funcs, var_dict, position)
    
    # now extract and arrange in the format of a sage matrix
    # each element in the tensor list will correspond to a row in the matrix
    
    row_list = []
    for index in A.tensors:
        row_list.append(A.tensors[index].data['data'].to_list())
        
    return matrix(row_list).inverse()            
                               
def evaluate(e_block, vec_list, tensor_dict):

    # first prepare the inputs
    premultiplied = [] # list of the premultiplied vectors
    for i in e_block.p_dict.keys():
        # loop over the premultiples
        if e_block.p_dict[i] == 0:
            # identity premultiple case
            premultiplied.append(vec_list[e_block.xs_dict[i][0] - 1]) # -1 for list indexing
        else:
            # now for the case when have to evalate at h_kx
            inputs = [] # these are the inputs for h_kx
            for j in e_block.xs_dict[i]:
                inputs.append(vec_list[j - 1])

            h_kx = tensor_dict[e_block.func2 + e_block.p_dict[i]*e_block.X] # use the strings to build key
            # h_kx is a SymbolicXYVectorTensor
            premultiplied.append(h_kx.evaluate(inputs))
    
    # speedup - only consider FKXJY
    # sort the s_string and apply that sort to the premultiplied
    
            
    return tensor_dict[e_block.func1 + "".join(sorted(e_block.s_string))].evaluate(
        [p for _,p in sorted(zip(e_block.s_string,premultiplied))])
    
def block_to_polynomial(y_dim, e_block, x_var_keys, var_dict, tensor_dict):
    
    order = e_block.n # the order of the polynomial terms
    
    # prepare list of polynomials, right now just the zero polynomial
    out = [0]*y_dim
    
    # build the basis vectors
    ei_vecs = {}
    link_dict = {}
    for i in range(1, len(x_var_keys) + 1):
        ei_vecs['e' + str(i)] = [int(i == j) for j in range(1, len(x_var_keys) + 1)]
        link_dict['e' + str(i)] = 'x' + str(i)
    
    # compute each term and add to out
    for combo in itertools.product(list(ei_vecs.keys()), repeat = order):
        #print([ei_vecs[key] for key in combo])
        # TODO - exploit symmetry for speedup
        coeffs = evaluate(e_block, [ei_vecs[key] for key in combo], tensor_dict) # evaluate at this combo of basis vectors
        term = prod([var_dict[link_dict[ei_key]] for ei_key in combo]) # xi*xj*xk^2 etc
        
        out = [a + b*term for a, b in zip(out, coeffs)] # update the output list
        
    return out    
    
def get_hkx_polynomial(funcs, k, x_dim, y_dim, var_dict, x_var_keys, tensor_dict, position):
    # note doesn't support constant c yet - but this doesn't appear so we are good
    
    if position != tensor_dict.position:
        raise(Exception('tensor dict position does not match position given'))
        
    if k == 0:
        # base case
        return [0]*y_dim
    
    # inductively compute the lower orders of hkx so they will be added to the tensor dict
    
    out = [0]*y_dim
    lower_order = get_hkx_polynomial(funcs, k-1, x_dim, y_dim, var_dict, x_var_keys, tensor_dict, position)
    print(lower_order)
    
    # now ready to solve for hkx
    # first get the expression we need
    
    b = ExpressionBlock() # defaults to just f
    e = Expression(blocks = [b]) # ready to differentiate
    for i in range(0,k):
        e = e.diff()
        
    # the last block is hkx
    for block in e.blocks[:-1]:
        #lprint(block)
        new_term = block_to_polynomial(y_dim, block, x_var_keys, var_dict, tensor_dict)
        out = [a + b/factorial(k) for a, b in zip(out, new_term)] # update the output list 
        # factorial since we are making a taylor expansion

        # TODO - is the factorial correct?
        
    # multiply by fyinv
    A = get_fy_inv(x_dim, y_dim, funcs, var_dict, position)
    out = list(-A*vector(out)) # using the derived formula
    
    # now convert to a tensor
    tensor = SymbolicXYVectorTensor(x_dim = x_dim , y_dim = y_dim, xy_order = 'X'*k, vec_length = y_dim)
    tensor.fill_from_functions(out, var_dict, position) # so avoids messy function evaluation
    tensor_dict['h' + 'X'*k] = tensor
    
    
    return [old + new for (old,new) in zip(lower_order, out)]
    
    
        
        
        
            

        
