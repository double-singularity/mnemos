pub fn cosine_similiarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    let lhs = dot_product(&a, &b)?;
    let rhs = magnitude(&a) * magnitude(&b);

    if rhs == 0.0 {
        return Err("magnitude is zero, cannot divide".to_string());
    }

    Ok(lhs / rhs)
}

fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Dimension mismatch: cannot dot product vectors of length {} and {}", 
            a.len(), 
            b.len()
        ));
    }

    let dot = a.iter()
               .zip(b.iter())
               .map(|(x, y)| x * y)
               .sum();

    Ok(dot)
}

fn magnitude(a: &[f32]) -> f32 {
    a.iter()
     .map(|x| x * x)
     .sum::<f32>()
     .sqrt()
}

#[cfg(test)] 
mod test {
    use super::*;

    #[test]
    fn test_dot_product() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0];

        assert!(dot_product(&v1, &v2).is_ok());
        assert_eq!(dot_product(&v1, &v2).unwrap(), 5.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(dot_product(&v1, &v2).is_err());
    }

    #[test]
    fn test_cosine_similiarity() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0];
        let result = cosine_similiarity(&v1, &v2).unwrap();
        
        assert!((result - 1.0).abs() < 1e-6);
    }
}