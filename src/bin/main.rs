use mnemos::storage::vector_store::VectorStore;
use mnemos::types::vector::Vector;
use mnemos::index::knn::knn_search;
use std::time::Instant;

fn main() {
    // 1. Setup: Create 10,000 random vectors
    let mut store = VectorStore::new(128);
    
    for i in 0..10_000 {
        let v = Vector { 
            id: i as u64, 
            values: (0..128).map(|_| rand::random::<f32>()).collect() 
        };
        // Use your insert method instead of .push()
        store.insert(v).expect("Failed to insert vector");
    }

    let query = Vector { 
        id: 99999, 
        values: (0..128).map(|_| rand::random::<f32>()).collect() 
    };

    // 2. Benchmark: Time the search
    let start = Instant::now();
    let results = knn_search( &store, &query, 10).unwrap_or_default();
    let duration = start.elapsed();

    // 3. Report
    println!("Found {} results in {:?}", results.len(), duration);
}