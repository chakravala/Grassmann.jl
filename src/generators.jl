
#   This file is part of Grassmann.jl. It is licensed under the GPL license
#   Grassmann Copyright (C) 2019 Michael Reed

@pure function labels(V::VectorSpace{N},label::Symbol=Symbol(pre[1]),dual::Symbol=Symbol(pre[2])) where N
    lab = string(label)
    io = IOBuffer()
    els = Array{Symbol,1}(undef,1<<N)
    els[1] = label
    icr = 1
    C = dualtype(V)
    C < 0 && (M = Int(N/2))
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            @inbounds sk = copy(set[k])
            if C < 0
                a = Int[]
                b = Int[]
                for j ∈ sk
                    push!(j ≤ M ? a : b, j)
                end
                b .-= M
                e = shift_indices(V,a)
                f = shift_indices(V,b)
                F = !isempty(f)
                if !(F && isempty(e))
                    @inbounds print(io,lab,a[1:min(9,end)]...)
                    for j ∈ 10:length(a)
                        print(io,subs[j])
                    end
                end
                if F
                    @inbounds print(io,string(dual),b[1:min(9,end)]...)
                    for j ∈ 10:length(b)
                        print(io,sups[j])
                    end
                end
            else
                print(io,C>0 ? string(dual) : lab)
                for j ∈ shift_indices(V,sk)
                    print(io,j≠0 ? (j>0 ? (j>9 ? (C>0 ? sups[j] : subs[j]) : j) : 'ϵ') : 'o')
                end
            end
            icr += 1
            @inbounds els[icr] = Symbol(String(take!(io)))
        end
    end
    return els
end

@pure function generate(V::VectorSpace{N}) where N
    exp = Basis{V}[Basis{V,0,zero(Bits)}()]
    for i ∈ 1:N
        set = combo(N,i)
        for k ∈ 1:length(set)
            @inbounds push!(exp,Basis{V,i,bit2int(indexbits(N,set[k]))}())
        end
    end
    return exp
end

export @basis, @basis_str, @dualbasis, @dualbasis_str, @mixedbasis, @mixedbasis_str

function basis(V::VectorSpace,sig::Symbol=vsn[1],label::Symbol=Symbol(pre[1]),dual::Symbol=Symbol(pre[2]))
    N = ndims(V)
    if N > algebra_limit
        Λ(V) # fill cache
        basis = generate(V)
        sym = labels(V,label,dual)
    else
        basis = Λ(V).b
        sym = labels(V,label,dual)
    end
    @inbounds exp = Expr[Expr(:(=),esc(sig),V),
        Expr(:(=),esc(label),basis[1])]
    for i ∈ 2:1<<N
        @inbounds push!(exp,Expr(:(=),esc(sym[i]),basis[i]))
    end
    return Expr(:block,exp...,Expr(:tuple,esc(sig),esc.(sym)...))
end

macro basis(q,sig=vsn[1],label=Symbol(pre[1]),dual=Symbol(pre[2]))
    basis(typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q),sig,label,dual)
end

macro basis_str(str)
    basis(VectorSpace(str))
end

macro dualbasis(q,sig=vsn[2],label=Symbol(pre[1]),dual=Symbol(pre[2]))
    basis((typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q))',sig,label,dual)
end

macro dualbasis_str(str)
    basis(VectorSpace(str)',vsn[2])
end

macro mixedbasis(q,sig=vsn[3],label=Symbol(pre[1]),dual=Symbol(pre[2]))
    V = typeof(q)∈(Symbol,Expr) ? (@eval(__module__,$q)) : VectorSpace(q)
    bases = basis(V⊕V',sig,label,dual)
    Expr(:block,bases,basis(V',vsn[2]),basis(V),bases.args[end])
end

macro mixedbasis_str(str)
    V = VectorSpace(str)
    bases = basis(V⊕V',vsn[3])
    Expr(:block,bases,basis(V',vsn[2]),basis(V),bases.args[end])
end

@pure @noinline function lookup_basis(V::VectorSpace,v::Symbol)::Union{SValue,Basis}
    vs = string(v)
    vt = vs[1:1]≠pre[1]
    Z=match(Regex("([$(pre[1])]([0-9a-vx-zA-VX-Z]+))?([$(pre[2])]([0-9a-zA-Z]+))?"),vs)
    ef = String[]
    for k ∈ (2,4)
        Z[k] ≠ nothing && push!(ef,Z[k])
    end
    length(ef) == 0 && (return zero(V))
    let W = V,fs=false
        C = dualtype(V)
        X = C≥0 && ndims(V)<4sizeof(Bits)+1
        X && (W = C>0 ? V'⊕V : V⊕V')
        V2 = (vt ⊻ (vt ? C≠0 : C>0)) ? V' : V
        L = length(ef) > 1
        M = X ? Int(ndims(W)/2) : ndims(W)
        m = ((!L) && vt && (C<0)) ? M : 0
        chars = (L || (Z[2] ≠ nothing)) ? alphanumv : alphanumw
        (es,e,et) = indexjoin([findfirst(isequal(ef[1][k]),chars) for k∈1:length(ef[1])].+m,C<0 ? V : V2)
        et && (return zero(V))
        d = if L
            (fs,f,ft) = indexjoin([findfirst(isequal(ef[2][k]),alphanumw) for k∈1:length(ef[2])].+M,W)
            ft && (return zero(V))
            Basis{W}([e;f])
        else
            Basis{V2}(e)
        end
        return (es⊻fs) ? SValue(-1,d) : d
    end
end
