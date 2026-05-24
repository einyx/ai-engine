use crate::arch::linear::LinearWeight;
use burn::tensor::{backend::Backend, Int, Tensor};

pub struct TokenEmbedding<B: Backend> {
    pub weight: Tensor<B, 2>, // [vocab, hidden]
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(weight: Tensor<B, 2>) -> Self {
        Self { weight }
    }

    /// `ids: [batch, seq]` -> `[batch, seq, hidden]`.
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = ids.dims();
        let hidden = self.weight.dims()[1];
        let flat = ids.reshape([batch * seq]);
        let gathered = self.weight.clone().select(0, flat); // [batch*seq, hidden]
        gathered.reshape([batch, seq, hidden])
    }
}

pub struct OutputProjection<B: Backend> {
    pub weight: LinearWeight<B>, // [hidden, vocab]
}

impl<B: Backend> OutputProjection<B> {
    pub fn new(weight: LinearWeight<B>) -> Self {
        Self { weight }
    }

    /// `x: [batch, seq, hidden]` -> `[batch, seq, vocab]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.weight.matmul(x)
    }
}
