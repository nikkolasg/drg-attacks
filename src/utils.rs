// msbd returns the most significant different bit index between the two given
// numbers. The index is 0-based counting from LSB to MSB.
// if u and v are equal, it returns an index higher than than the bitsize.
// TODO: make that generic for xorable values ?
pub fn msbd(u: usize, v: usize) -> usize {
    let bitsize = node_bitsize();
    let xor = u ^ v;
    if xor == 0 {
        return bitsize + 1;
        // FIXME: The +1 seems unnecessary, for a 0-based index
        // the size is already out of bounds.
    }

    let mut idx = bitsize - 1; // index of the different bit
    loop {
        if xor & (1 << idx) > 0 {
            return idx;
        }
        idx -= 1;
    }
}

pub fn node_bitsize() -> usize {
    std::mem::size_of::<usize>() * 8
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_msbd() {
        assert_eq!(msbd(4, 2), 2);
        assert_eq!(msbd(0, 2), 1);
        assert_eq!(msbd(0, 1), 0);
        assert_eq!(msbd(2, 3), 0);
    }
}
