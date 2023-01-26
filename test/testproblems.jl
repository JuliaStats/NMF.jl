# Hans Laurberg, Mads Græsbøll Christensen, Mark D. Plumbley, Lars Kai Hansen,
# Søren Holdt Jensen, "Theorems on Positive Data: On the Uniqueness of NMF",
# Computational Intelligence and Neuroscience, vol. 2008, Article ID 764206, 9
# pages, 2008. https://doi.org/10.1155/2008/764206
# Example 3 from Section 5, Eqs. 9 (for α = 0.1 or α = 0.3, the NMF is unique up to scaling)
function laurberg6x3(α)
    H = [α 1 1 α 0 0
         1 α 0 0 α 1
         0 0 α 1 1 α]
    W = H'
    X = W*H
    return X, Matrix(W), H
end
