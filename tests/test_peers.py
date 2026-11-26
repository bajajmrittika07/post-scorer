"""Peers: hand-collected follower counts are parsed as typed on the profile."""
from post_scorer.peers import parse_count, export_peers


def test_parse_count_handles_profile_formats():
    assert parse_count("111.7M") == 111_700_000
    assert parse_count("44.1k") == 44_100
    assert parse_count("3M") == 3_000_000
    assert parse_count("1,234") == 1234
    assert parse_count("") is None


def test_export_skips_title_line_blanks_tiktok_and_handles(tmp_path):
    p = tmp_path / "peers.csv"
    p.write_text("my_file_title\nname,party,role,platform,handle,followers,checked_on\n"
                 "A,Republican,President,X,@a,10M,2026-11-26\n"
                 "A,Republican,President,TikTok,,,2026-11-26\n"
                 "B,Democrat,Mayor,Truth Social,,,2026-11-26\n"
                 "B,Democrat,Mayor,Instagram,@b,302k,2026-11-26\n")
    rows = export_peers(str(p))
    assert [(r["name"], r["platform"], r["followers"]) for r in rows] == [("A", "X", 10_000_000), ("B", "Instagram", 302_000)]
    assert all("handle" not in r for r in rows)
