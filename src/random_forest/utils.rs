
use rand::prelude::*;




pub fn split_dataset(x: &mut Vec<Vec<f32>>, y: &mut Vec<String>, train_size: f32) -> (Vec<Vec<f32>>,  Vec<String>, Vec<Vec<f32>>,Vec<String>) {
    let n_train = (x.len() as f32 * train_size) as usize;
    let n_test = x.len() - n_train;
    
    let mut x_test: Vec<Vec<f32>> = vec![];
    let mut x_train: Vec<Vec<f32>> = vec![];
    let mut y_test: Vec<String> = vec![];
    let mut y_train: Vec<String> = vec![];
    let mut idxs: Vec<usize> = (0..x.len()).collect();


    let mut rng = rand::rngs::StdRng::seed_from_u64(41);

    idxs.shuffle(&mut rng);

    
    
    for (i, &idx) in idxs.iter().enumerate(){
        let xx: Vec<f32> = x[idx].clone();
        let yy: String = y[idx].clone();
        
        if i < n_train {
            x_train.push(xx);
            y_train.push(yy);
        }
        else {
            x_test.push(xx);
            y_test.push(yy);
        }

        
    }
    return (x_train, y_train, x_test, y_test);
}

pub fn accuracy_per_label(y: &Vec<String>, y_hat: &Vec<String>) -> Vec<f32> {
    let mut acc: Vec<f32> = vec![];

    let labels = unique_vals(&y);


    for i in 0..labels.len() {
        let mut c = 0.0;
        for j in 0..y_hat.len() {
            if y[j] == labels[i] {
                if y_hat[j] == y[j] {
                    c = c + 1.0;
                }
            }
        }

        let total_labels = count_vals(&y, labels[i].clone());
        let label_acc = c / (total_labels as f32);
        acc.push(label_acc);
    }

    acc
}

pub fn count_vals(arr: &Vec<String>, label: String) -> usize {
    let mut c = 0;
    for el in arr.iter() {
        if el == &label {
            c = c + 1;
        }
    }
    
    c
}

pub fn unique_vals(arr: &Vec<String>) -> Vec<String> {
    let mut u_vals: Vec<String> = vec![];
    for el in arr.iter() {
        if !u_vals.contains(&el) {
            u_vals.push(el.to_string());
        }
    }
    u_vals
}

pub fn unique_vals_f32(arr: &Vec<f32>) -> Vec<f32> {
    let mut u_vals: Vec<f32> = vec![];
    for el in arr.iter() {
        if !u_vals.contains(&el) {
            u_vals.push(*el);
        }
    }
    println!("antes: {:?}", u_vals);
    u_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("despues: {:?}", u_vals);
    let returnded = u_vals.clone();

    return returnded;
}


pub fn get_column(matrix: &Vec<Vec<f32>>, col: usize) -> Vec<f32>{
    let mut column: Vec<f32> = vec![];
    for row in matrix.iter() {
        for (j, &colu) in row.iter().enumerate() {
            if j == col {
                //println!("{} {}", j, col);
                column.push(colu);
            }
        }
    }
    column
}
