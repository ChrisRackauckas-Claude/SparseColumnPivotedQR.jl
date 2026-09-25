module SparseColumnPivotedQRAMDExt

using SparseColumnPivotedQR: SparseColumnPivotedQR
using AMD: colamd
using SparseArrays: SparseMatrixCSC

# Flag the host module so `:default` ordering resolves to `:amd` and so the
# `:amd` opt-in doesn't error. Set on extension load, never cleared.
function __init__()
    SparseColumnPivotedQR._AMD_EXT_LOADED[] = true
    return nothing
end

function SparseColumnPivotedQR._amd_colperm(
        rowptr::Vector{Int}, colval::Vector{Int},
        m::Int, n::Int
    )
    nnz_total = length(colval)
    colcounts = zeros(Int, n)
    @inbounds for q in 1:nnz_total
        colcounts[colval[q]] += 1
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] = colptr[j] + colcounts[j]
    end

    rowidx = Vector{Int}(undef, nnz_total)
    work = copy(colptr)
    @inbounds for i in 1:m
        r1 = rowptr[i]; r2 = rowptr[i + 1] - 1
        for q in r1:r2
            j = colval[q]
            rowidx[work[j]] = i
            work[j] += 1
        end
    end

    pattern = SparseMatrixCSC(m, n, colptr, rowidx, ones(Bool, nnz_total))
    return Int.(colamd(pattern))
end

end # module
