use std::collections::HashMap;
use crate::types::vector::Vector;

pub struct VectorStore {
    pub vectors: HashMap<u64, Vector>,
    pub dimension: usize,
}

impl VectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            dimension,
        }
    }

    pub fn insert(&mut self, vector: Vector) -> Result<(), String> {
        if vector.values.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}", 
                self.dimension, 
                vector.values.len()
            ));
        }

        self.vectors.insert(vector.id, vector);

        Ok(())
    }

    pub fn get(&self, id: &u64) -> Option<&Vector> {
        self.vectors.get(id)
    }
}