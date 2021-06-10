use crate::graph::Edge;

// msbd returns the most significant different bit index between the two given
// numbers. The index is 0-based counting from LSB to MSB.
// if u and v are equal, it returns an index higher than than the bitsize.
// TODO: make that generic for xorable values ?
// FIXME: Make the return type a bit position (to help check for overflows with
// the node type).
pub fn msbd(edge: &Edge) -> usize {
    let bitsize = node_bitsize();
    let xor = edge.parent ^ edge.child;
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

pub fn to_hex_string(bytes: &[u8]) -> String {
    let strs: Vec<String> = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    strs.join("")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_msbd() {
        assert_eq!(msbd(&Edge::new(2, 4)), 2);
        assert_eq!(msbd(&Edge::new(0, 2)), 1);
        assert_eq!(msbd(&Edge::new(0, 1)), 0);
        assert_eq!(msbd(&Edge::new(2, 3)), 0);
    }
}
