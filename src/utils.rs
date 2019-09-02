// msbd returns the most significant different bit index between the two given
// numbers. The index is 0-based.
// if u and v are equal, it returns an index higher than than the bitsize.
// TODO: make that generic for xorable values ?
pub fn msbd(u: usize, v: usize) -> usize {
    let bitsize = node_bitsize();
    let xor = u ^ v;
    let mut idx = 0; // index of the different bit
    while idx < bitsize {
        let diff = xor >> (bitsize - 1 - idx);
        if diff == 1 {
            return idx;
        }
        idx += 1;
    }
    return bitsize + 1;
}

pub fn node_bitsize() -> usize {
    std::mem::size_of::<usize>() * 8
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_msbd() {
        assert_eq!(msbd(4, 2), 61);
        assert_eq!(msbd(0, 2), 62);
        assert_eq!(msbd(0, 1), 63);
        assert_eq!(msbd(2, 3), 63);
    }
}
