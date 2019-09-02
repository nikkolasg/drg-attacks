// msbd returns the most significant different bit index between the two given
// numbers. The index is 0-based.
// if u and v are equal, it returns an index higher than than the bitsize.
// TODO: make that generic for xorable values ?
pub fn msbd(u: usize, v: usize) -> u8 {
    let bitsize = std::mem::size_of::<usize>() * 8;
    println!("bitsize = {}", bitsize);
    let xor = u ^ v;
    let mut idx = 0; // index of the different bit
    while idx < bitsize {
        let diff = xor >> (bitsize - 1 - idx);
        if diff == 1 {
            return idx as u8;
        }
        idx += 1;
    }
    return (bitsize + 1) as u8;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_msbd() {
        assert_eq!(msbd(4, 2), 61);
    }
}
