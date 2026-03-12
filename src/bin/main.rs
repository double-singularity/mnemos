use mnemos::storage::vector_store::VectorStore;
use mnemos::types::vector::Vector;

fn main() {
    let mut store = VectorStore::new(3);
    store.insert(
        "vec1".to_string(), 
        Vector { id: 1, values: vec![1.0, 0.0, 0.0]}
    ).unwrap();

    store.insert(
        "vec2".to_string(), 
        Vector { id: 2, values: vec![0.0, 1.0, 0.0]}
    ).unwrap();

    if let Some(v) = store.get("vec1") {
        println!("{:?}", v.values);
    }
}