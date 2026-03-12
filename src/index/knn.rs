use crate::storage::vector_store::VectorStore;
use crate::types::vector::Vector;
use crate::distance::metrics::cosine_similiarity;

pub fn knn_search<'a>(
    store:&'a VectorStore, 
    query: &Vector, 
    k: usize
) -> Option<Vec<(&'a u64, f32)>> {   
    if k == 0 || store.vectors.is_empty() {
        return None;
    }

    let mut results: Vec<(&'a u64, f32)> = store.vectors.iter()
        .filter_map(|(id, vector)| {
            cosine_similiarity(&query.values, &vector.values).ok().map(|score| (id, score))
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Some(results.into_iter().take(k).collect())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::vector_store::VectorStore;
    use crate::types::vector::Vector;

    fn build_store() -> VectorStore {
        let mut store = VectorStore::new(3); 
        store.insert(Vector { id: 1, values: vec![1.0, 0.0, 0.0] }).unwrap();
        store.insert(Vector { id: 2, values: vec![0.0, 1.0, 0.0] }).unwrap();
        store.insert(Vector { id: 3, values: vec![0.0, 0.0, 1.0] }).unwrap();
        store
    }

    #[test]
    fn test_knn_normal() {
        let store = build_store();
        let query = Vector { id: 0, values: vec![1.0, 0.0, 0.0] };

        let neighbors = knn_search(&store, &query, 2).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, &1); 
    }

    #[test]
    fn test_knn_k_zero() {
        let store = build_store();
        let query = Vector { id: 0, values: vec![1.0, 0.0, 0.0] };

        let neighbors = knn_search(&store, &query, 0);
        assert!(neighbors.is_none());
    }

    #[test]
    fn test_knn_empty_store() {
        let store = VectorStore::new(3);
        let query = Vector { id: 0, values: vec![1.0, 0.0, 0.0] };

        let neighbors = knn_search(&store, &query, 3);
        assert!(neighbors.is_none()); 
    }

    #[test]
    fn test_knn_k_greater_than_store() {
        let store = build_store();
        let query = Vector { id: 0, values: vec![1.0, 0.0, 0.0] };

        let neighbors = knn_search(&store, &query, 10).unwrap();
        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_knn_all_equal_scores() {
        let mut store = VectorStore::new(3);
        let query = Vector { id: 0, values: vec![1.0, 1.0, 1.0] };

        for i in 0..5 {
            store.insert(Vector { id: i, values: vec![1.0, 1.0, 1.0] }).unwrap();
        }

        let neighbors = knn_search(&store, &query, 3).unwrap();
        assert_eq!(neighbors.len(), 3);

        let first_score = neighbors[0].1;
        for (_, score) in neighbors.iter() {
            assert!((score - first_score).abs() < 1e-6);
        }
    }
}