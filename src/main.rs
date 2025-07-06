use image::RgbImage;
use std::path::Path;
use std::time::Instant;

const ITERATIONS: u32 = 100;
const WIDTH: u32 = 3500;
const HEIGHT: u32 = 2000;
const BUFFER_SIZE: usize = (WIDTH * HEIGHT * 3) as usize;


struct Complex {
    real: f64,
    im: f64,
}

impl Complex {
    fn square(&mut self) {
	let prev_real = self.real;
	self.real = self.real * self.real - self.im * self.im;
	self.im = 2.0 * prev_real * self.im;
    }

    fn plus(&mut self, other: &Complex) {
	self.real = self.real + other.real;
	self.im = self.im + other.im;
    }

    fn abs(&self) -> f64 {
	(self.real * self.real + self.im * self.im).sqrt()
    }
}

fn main() {
    // sequential
    println!("Starting Sequential algorithm");
    let mut buffer: Vec<u8> = vec![0; BUFFER_SIZE];
    let now = Instant::now();
    for x in 0..WIDTH {
	for y in 0..HEIGHT {
	    let c = Complex {
		real: x as f64 / 1000.0 - 2.5,
		im: y as f64 / 1000.0 - 1.0,
	    };
	    let mut z = Complex {
		real: 0.0,
		im: 0.0,
	    };
	    let mut i = 0;
	    
	    while i < ITERATIONS && z.abs() < 2.0 {
		z.square();
		z.plus(&c);
		i = i + 1;
	    }
	    if i == ITERATIONS {
		let index = ((y * WIDTH + x) * 3) as usize;
		buffer[index] = 255;
		buffer[index+1] = 255;
		buffer[index+2] = 255;
	    }
	}
    }
    let elapsed = now.elapsed();
    println!("Time for Sequential algorithm: {}ms", elapsed.as_millis());    
    save_fractal(buffer, &Path::new("sequential.png"));

    // optimized sequential
    println!("Starting Optimized Sequential algorithm");
    let mut buffer: Vec<u8> = vec![0; BUFFER_SIZE];
    let now = Instant::now();
    for x in 0..WIDTH {
	for y in 0..HEIGHT {
	    let cx = x as f64 / 1000.0 - 2.5;
	    let cy = y as f64 / 1000.0 - 1.0;
	    let mut zx = 0.0;
	    let mut zy = 0.0;
	    let mut i = 0;
	    
	    while i < ITERATIONS && zx*zx + zy*zy < 4.0 {
		let prev_zx = zx;
		zx = zx*zx - zy*zy + cx;
		zy = 2.0 * prev_zx * zy + cy;
		i = i + 1;
	    }
	    if i == ITERATIONS {
		let index = ((y * WIDTH + x) * 3) as usize;
		buffer[index] = 255;
		buffer[index+1] = 255;
		buffer[index+2] = 255;
	    }
	}
    }
    let elapsed = now.elapsed();
    println!("Time for Optimized Sequential algorithm: {}ms", elapsed.as_millis());    
    save_fractal(buffer, &Path::new("opti-sequential.png"));

}


fn save_fractal(buffer: Vec<u8>, path: &Path) {
    let img = RgbImage::from_raw(WIDTH, HEIGHT, buffer).unwrap();

    img.save(path).unwrap();
}
