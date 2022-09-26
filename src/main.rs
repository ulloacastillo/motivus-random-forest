

use rand::prelude::*;
use std::fs::File;
use std::io::Read;
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

mod utils;
mod random_forest;
mod dtree;

// #[derive(Debug, Deserialize, Serialize)]
// struct Data {
//     max_depth: usize,
//     min_samples_split: usize,
//     seed: usize,
//     x: Vec<Vec<f32>>,
//     y: Vec<String>
// }
#[derive(Debug, Deserialize, Serialize)]
struct Data {
    n_trees: usize,
    min_samples_split: usize,
    max_depth: usize,
    n_feats: usize,
    seed: u64,
}


fn split_dataset(x: &mut Vec<Vec<f32>>, y: &mut Vec<String>, train_size: f32) -> (Vec<Vec<f32>>,  Vec<String>, Vec<Vec<f32>>,Vec<String>) {
    let n_train = (x.len() as f32 * train_size) as usize;
    
    
    let mut x_test: Vec<Vec<f32>> = vec![];
    let mut x_train: Vec<Vec<f32>> = vec![];
    let mut y_test: Vec<String> = vec![];
    let mut y_train: Vec<String> = vec![];
    let mut idxs: Vec<usize> = (0..x.len()).collect();


    let mut rng = rand::thread_rng();

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

fn accuracy_per_label(y: &Vec<String>, y_hat: &Vec<String>) -> Vec<f32> {
    let mut acc: Vec<f32> = vec![];

    let labels = utils::unique_vals(&y);


    for i in 0..labels.len() {
        let mut c = 0.0;
        for j in 0..y_hat.len() {
            if y[j] == labels[i] {
                if y_hat[j] == y[j] {
                    c = c + 1.0;
                }
            }
        }

        let total_labels = utils::count_vals(&y, labels[i].clone());
        let label_acc = c / (total_labels as f32);
        acc.push(label_acc);
    }

    acc
}



fn main()  {  //-> JsValue
    println!("Hello, world!");
   
    let mut file = File::open("./iris.csv").expect("No se pudo abrir el arcivo");

    let mut content = String::new();

    file.read_to_string(&mut content).expect("No se pudo leer el archivo");


    //println!("{}", content);


    let mut _reader = csv::Reader::from_reader(content.as_bytes());
    

    //let content: String = text.into_serde().unwrap();


    let values: Vec<String> = content.split('\n').map(str::to_string).collect();
    let mut y: Vec<String> = vec![];
    let mut x: Vec<Vec<f32>> = vec![];

    // Pretty-print the results.
    let _xs: Vec<String> = values[0].split(',').map(str::to_string).collect();
    
    for _row in values.iter() {
        let xs: Vec<String> = _row.split(',').map(str::to_string).collect();
        
        let mut aux: Vec<f32> = vec![];
        for i in 0..&xs.len()-1{
            aux.push(xs[i].parse::<f32>().unwrap());
        }
        let a = &xs.len() -1;
        let b = &xs[a];
        &y.push(b.to_string());
        &x.push(aux);
    }
    

    let (x_train, y_train, _x_test, _y_test) = utils::split_dataset(&mut x, &mut y, 0.8);
    let mut rf = random_forest::RandomForest::new(1, 3, 3, 4, 41);
    rf.fit(&x_train, &y_train);

    let y_pred = rf.predict(&x_train);
        
    let percent = random_forest::accuracy(&y_pred, &y_train);

    println!("{}", percent);
    
    /*
    let task: String = task_type.into_serde().unwrap();

    if "predict".to_string().eq(&task) {
        let mut rf: random_forest::RandomForest = json.into_serde().unwrap();
        let x_test: Vec<Vec<f32>> = train_dataset.into_serde().unwrap();
        
        let y_pred = rf.predict(&x_test);
        JsValue::from_serde(&y_pred).unwrap()
    }

    else {
        let params: Data = json.into_serde().unwrap();

        let mut x: Vec<Vec<f32>> = train_dataset.into_serde().unwrap();
        let mut y: Vec<String> = train_labels.into_serde().unwrap();

        let seed: u64 = params.seed;

        let (x_train, y_train, _x_test, _y_test) = utils::split_dataset(&mut x, &mut y, 0.8);

        // Parameters: n_trees, min_samples_split, max_depth, n_feats
        let mut rf = random_forest::RandomForest::new(100, 3, 3, 4, seed);
        
        rf.fit(&x_train, &y_train);

        let y_pred = rf.predict(&x_train);
        
        let percent = random_forest::accuracy(&y_pred, &y_train);
        
        JsValue::from_serde(&vec![y_pred, y_train]).unwrap()
    }


    //println!("Accuracy: {}", random_forest::accuracy(&y_test, &y_pred));



    //println!("{:?}", random_forest::accuracy(&y_test, &y_pred));

    
    */
    
    
    
    
}


