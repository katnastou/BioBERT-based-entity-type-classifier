#!/usr/bin/perl -w

use strict;
use POSIX;

my %seen = ();
open IN, "< tagger-all-dictionary/excluded_documents.txt";
while (<IN>) {
	s/\r?\n//;
	$seen{$_} = 1;
}
close IN;

open IN, "gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` |";
open OUT, "> database_documents.tsv";
while (<IN>) {
	s/\r?\n//;
	s/\\/\\\\/g;
	my ($keys, $authors, $publication, $year, @text) = split /\t/;
	next unless $keys =~ /^PMID:([0-9]+)/;
	my $document = $1;
	next if exists $seen{$document};
	$seen{$document} = 1;
	next unless defined $authors and defined $publication and defined $year;
	print OUT $document, "\t", $keys, "\t", $authors, "\t", $publication, "\t", $year, "\t", join("\\t", @text), "\n";
}
close IN;
close OUT;

close STDERR;
close STDOUT;
POSIX::_exit(0);
