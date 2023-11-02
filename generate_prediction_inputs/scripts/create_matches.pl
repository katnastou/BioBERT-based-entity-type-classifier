#!/usr/bin/perl -w

use strict;
use POSIX;

my %serial_type_identifier = ();
open IN, "<", $ARGV[0];
#open IN, "< dictionary-files-tagger-STRINGv12/all_entities.tsv";
while (<IN>) {
	s/\r?\n//;
	my ($serial, $type, $identifier) = split /\t/;
	$serial_type_identifier{$serial} = $type."\t".$identifier;
}
close IN;
print ("entities file read\n");
open IN, "<", $ARGV[1];
open OUT, ">", $ARGV[2];
# open IN, "< no-block/all_matches.tsv";
# open OUT, "> no-block/database_matches.tsv";

while (<IN>) {
	s/\r?\n//;
	my ($document, undef, undef, $start, $stop, undef, $type, $serial) = split /\t/;
	print OUT $document, "\t", $start, "\t", $stop, "\t", $serial_type_identifier{$serial}, "\n" if exists $serial_type_identifier{$serial};
}
close IN;
close OUT;

close STDERR;
close STDOUT;
POSIX::_exit(0);
