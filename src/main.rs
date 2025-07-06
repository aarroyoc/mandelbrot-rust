use image::{RgbImage, Rgb};

const ITERATIONS: u32 = 50;
const WIDTH: u32 = 3500;
const HEIGHT: u32 = 2000;
const BUFFER_SIZE: usize = (WIDTH * HEIGHT * 3) as usize

fn main() {
    // sequential

    let buffer: Vec<u8> = vec![0; BUFFER_SIZE];

    for x in 0..WIDTH {
	for y in 0..HEIGHT {
	    let cx = x / 1000 - 2.5;
	    let cy = y / 1000 - 1.0;

	    let mut zx = 0;
	    let mut zy = 0;

	    let mut i = 0;
	    
	    while i < ITERATIONS || (zx * zx + zy * zy) < 4 {
		let real = zx * zx - zy * zy + cx;
		let img = 2 * zx * zy + cy;

		i = i + 1
	    }
		
}


fn save_fractal(buffer: Vec<u8>, path: Path) {
    let img = RgbImage::from_raw(WIDTH, HEIGHT, buffer).unwrap();

    img.save(path).unwrap();
}
