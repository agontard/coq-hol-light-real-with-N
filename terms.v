Require Import Stdlib.NArith.BinNat mathcomp.classical.classical_sets HOLLight_Real_With_N.type HOLLight_Real_With_N.mappings.
Definition _FALSITY_ : Prop := False.
Lemma _FALSITY__def : _FALSITY_ = False.
Proof. exact (REFL _FALSITY_). Qed.
Definition o {A B C : Type'} : (B -> C) -> (A -> B) -> A -> C := fun f : B -> C => fun g : A -> B => fun x : A => f (g x).
Lemma o_def {A B C : Type'} : (@o A B C) = (fun f : B -> C => fun g : A -> B => fun x : A => f (g x)).
Proof. exact (REFL (@o A B C)). Qed.
Definition I {A : Type'} : A -> A := fun x : A => x.
Lemma I_def {A : Type'} : (@I A) = (fun x : A => x).
Proof. exact (REFL (@I A)). Qed.
Definition hashek : Prop := True.
Lemma hashek_def : hashek = True.
Proof. exact (REFL hashek). Qed.
Definition LET {A B : Type'} : (A -> B) -> A -> B := fun f : A -> B => fun x : A => f x.
Lemma LET_def {A B : Type'} : (@LET A B) = (fun f : A -> B => fun x : A => f x).
Proof. exact (REFL (@LET A B)). Qed.
Definition LET_END {A : Type'} : A -> A := fun t : A => t.
Lemma LET_END_def {A : Type'} : (@LET_END A) = (fun t : A => t).
Proof. exact (REFL (@LET_END A)). Qed.
Definition GABS {A : Type'} : (A -> Prop) -> A := fun P : A -> Prop => @ε A P.
Lemma GABS_def {A : Type'} : (@GABS A) = (fun P : A -> Prop => @ε A P).
Proof. exact (REFL (@GABS A)). Qed.
Definition _SEQPATTERN {A B : Type'} : (A -> B -> Prop) -> (A -> B -> Prop) -> A -> B -> Prop := fun r : A -> B -> Prop => fun s : A -> B -> Prop => fun x : A => @COND (B -> Prop) (exists y : B, r x y) (r x) (s x).
Lemma _SEQPATTERN_def {A B : Type'} : (@_SEQPATTERN A B) = (fun r : A -> B -> Prop => fun s : A -> B -> Prop => fun x : A => @COND (B -> Prop) (exists y : B, r x y) (r x) (s x)).
Proof. exact (REFL (@_SEQPATTERN A B)). Qed.
Definition _UNGUARDED_PATTERN : Prop -> Prop -> Prop := fun p : Prop => fun r : Prop => p /\ r.
Lemma _UNGUARDED_PATTERN_def : _UNGUARDED_PATTERN = (fun p : Prop => fun r : Prop => p /\ r).
Proof. exact (REFL _UNGUARDED_PATTERN). Qed.
Definition _GUARDED_PATTERN : Prop -> Prop -> Prop -> Prop := fun p : Prop => fun g : Prop => fun r : Prop => p /\ (g /\ r).
Lemma _GUARDED_PATTERN_def : _GUARDED_PATTERN = (fun p : Prop => fun g : Prop => fun r : Prop => p /\ (g /\ r)).
Proof. exact (REFL _GUARDED_PATTERN). Qed.
Definition _MATCH {A B : Type'} : A -> (A -> B -> Prop) -> B := fun e : A => fun r : A -> B -> Prop => @COND B (@ex1 B (r e)) (@ε B (r e)) (@ε B (fun z : B => False)).
Lemma _MATCH_def {A B : Type'} : (@_MATCH A B) = (fun e : A => fun r : A -> B -> Prop => @COND B (@ex1 B (r e)) (@ε B (r e)) (@ε B (fun z : B => False))).
Proof. exact (REFL (@_MATCH A B)). Qed.
Definition _FUNCTION {A B : Type'} : (A -> B -> Prop) -> A -> B := fun r : A -> B -> Prop => fun x : A => @COND B (@ex1 B (r x)) (@ε B (r x)) (@ε B (fun z : B => False)).
Lemma _FUNCTION_def {A B : Type'} : (@_FUNCTION A B) = (fun r : A -> B -> Prop => fun x : A => @COND B (@ex1 B (r x)) (@ε B (r x)) (@ε B (fun z : B => False))).
Proof. exact (REFL (@_FUNCTION A B)). Qed.
Definition CURRY {A B C : Type'} : ((prod A B) -> C) -> A -> B -> C := fun _1283 : (prod A B) -> C => fun _1284 : A => fun _1285 : B => _1283 (@pair A B _1284 _1285).
Lemma CURRY_def {A B C : Type'} : (@CURRY A B C) = (fun _1283 : (prod A B) -> C => fun _1284 : A => fun _1285 : B => _1283 (@pair A B _1284 _1285)).
Proof. exact (REFL (@CURRY A B C)). Qed.
Definition UNCURRY {A B C : Type'} : (A -> B -> C) -> (prod A B) -> C := fun _1304 : A -> B -> C => fun _1305 : prod A B => _1304 (@fst A B _1305) (@snd A B _1305).
Lemma UNCURRY_def {A B C : Type'} : (@UNCURRY A B C) = (fun _1304 : A -> B -> C => fun _1305 : prod A B => _1304 (@fst A B _1305) (@snd A B _1305)).
Proof. exact (REFL (@UNCURRY A B C)). Qed.
Definition PASSOC {A B C D : Type'} : ((prod (prod A B) C) -> D) -> (prod A (prod B C)) -> D := fun _1321 : (prod (prod A B) C) -> D => fun _1322 : prod A (prod B C) => _1321 (@pair (prod A B) C (@pair A B (@fst A (prod B C) _1322) (@fst B C (@snd A (prod B C) _1322))) (@snd B C (@snd A (prod B C) _1322))).
Lemma PASSOC_def {A B C D : Type'} : (@PASSOC A B C D) = (fun _1321 : (prod (prod A B) C) -> D => fun _1322 : prod A (prod B C) => _1321 (@pair (prod A B) C (@pair A B (@fst A (prod B C) _1322) (@fst B C (@snd A (prod B C) _1322))) (@snd B C (@snd A (prod B C) _1322)))).
Proof. exact (REFL (@PASSOC A B C D)). Qed.
Definition minimal : (N -> Prop) -> N := fun _6536 : N -> Prop => @ε N (fun n : N => (_6536 n) /\ (forall m : N, (N.lt m n) -> ~ (_6536 m))).
Lemma minimal_def : minimal = (fun _6536 : N -> Prop => @ε N (fun n : N => (_6536 n) /\ (forall m : N, (N.lt m n) -> ~ (_6536 m)))).
Proof. exact (REFL minimal). Qed.
Definition MEASURE {A : Type'} : (A -> N) -> A -> A -> Prop := fun _8094 : A -> N => fun x : A => fun y : A => N.lt (_8094 x) (_8094 y).
Lemma MEASURE_def {A : Type'} : (@MEASURE A) = (fun _8094 : A -> N => fun x : A => fun y : A => N.lt (_8094 x) (_8094 y)).
Proof. exact (REFL (@MEASURE A)). Qed.
Definition NUMPAIR : N -> N -> N := fun _17487 : N => fun _17488 : N => N.mul (N.pow (NUMERAL (BIT0 (BIT1 N0))) _17487) (N.add (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17488) (NUMERAL (BIT1 N0))).
Lemma NUMPAIR_def : NUMPAIR = (fun _17487 : N => fun _17488 : N => N.mul (N.pow (NUMERAL (BIT0 (BIT1 N0))) _17487) (N.add (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17488) (NUMERAL (BIT1 N0)))).
Proof. exact (REFL NUMPAIR). Qed.
Definition NUMFST : N -> N := @ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun X : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17503 : prod N (prod N (prod N (prod N (prod N N)))), exists Y : N -> N, forall x : N, forall y : N, ((X _17503 (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0))))))))))))).
Lemma NUMFST_def : NUMFST = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun X : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17503 : prod N (prod N (prod N (prod N (prod N N)))), exists Y : N -> N, forall x : N, forall y : N, ((X _17503 (NUMPAIR x y)) = x) /\ ((Y (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))))).
Proof. exact (REFL NUMFST). Qed.
Definition NUMSND : N -> N := @ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17504 : prod N (prod N (prod N (prod N (prod N N)))), forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y _17504 (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0))))))))))))).
Lemma NUMSND_def : NUMSND = (@ε ((prod N (prod N (prod N (prod N (prod N N))))) -> N -> N) (fun Y : (prod N (prod N (prod N (prod N (prod N N))))) -> N -> N => forall _17504 : prod N (prod N (prod N (prod N (prod N N)))), forall x : N, forall y : N, ((NUMFST (NUMPAIR x y)) = x) /\ ((Y _17504 (NUMPAIR x y)) = y)) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT0 (BIT0 (BIT1 N0)))))))))))))).
Proof. exact (REFL NUMSND). Qed.
Definition NUMSUM : Prop -> N -> N := fun _17505 : Prop => fun _17506 : N => @COND N _17505 (N.succ (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17506)) (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17506).
Lemma NUMSUM_def : NUMSUM = (fun _17505 : Prop => fun _17506 : N => @COND N _17505 (N.succ (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17506)) (N.mul (NUMERAL (BIT0 (BIT1 N0))) _17506)).
Proof. exact (REFL NUMSUM). Qed.
Definition INJN {A : Type'} : N -> N -> A -> Prop := fun _17537 : N => fun n : N => fun a : A => n = _17537.
Lemma INJN_def {A : Type'} : (@INJN A) = (fun _17537 : N => fun n : N => fun a : A => n = _17537).
Proof. exact (REFL (@INJN A)). Qed.
Definition INJA {A : Type'} : A -> N -> A -> Prop := fun _17542 : A => fun n : N => fun b : A => b = _17542.
Lemma INJA_def {A : Type'} : (@INJA A) = (fun _17542 : A => fun n : N => fun b : A => b = _17542).
Proof. exact (REFL (@INJA A)). Qed.
Definition INJF {A : Type'} : (N -> N -> A -> Prop) -> N -> A -> Prop := fun _17549 : N -> N -> A -> Prop => fun n : N => _17549 (NUMFST n) (NUMSND n).
Lemma INJF_def {A : Type'} : (@INJF A) = (fun _17549 : N -> N -> A -> Prop => fun n : N => _17549 (NUMFST n) (NUMSND n)).
Proof. exact (REFL (@INJF A)). Qed.
Definition INJP {A : Type'} : (N -> A -> Prop) -> (N -> A -> Prop) -> N -> A -> Prop := fun _17554 : N -> A -> Prop => fun _17555 : N -> A -> Prop => fun n : N => fun a : A => @COND Prop (NUMLEFT n) (_17554 (NUMRIGHT n) a) (_17555 (NUMRIGHT n) a).
Lemma INJP_def {A : Type'} : (@INJP A) = (fun _17554 : N -> A -> Prop => fun _17555 : N -> A -> Prop => fun n : N => fun a : A => @COND Prop (NUMLEFT n) (_17554 (NUMRIGHT n) a) (_17555 (NUMRIGHT n) a)).
Proof. exact (REFL (@INJP A)). Qed.
Definition ZCONSTR {A : Type'} : N -> A -> (N -> N -> A -> Prop) -> N -> A -> Prop := fun _17566 : N => fun _17567 : A => fun _17568 : N -> N -> A -> Prop => @INJP A (@INJN A (N.succ _17566)) (@INJP A (@INJA A _17567) (@INJF A _17568)).
Lemma ZCONSTR_def {A : Type'} : (@ZCONSTR A) = (fun _17566 : N => fun _17567 : A => fun _17568 : N -> N -> A -> Prop => @INJP A (@INJN A (N.succ _17566)) (@INJP A (@INJA A _17567) (@INJF A _17568))).
Proof. exact (REFL (@ZCONSTR A)). Qed.
Definition ZBOT {A : Type'} : N -> A -> Prop := @INJP A (@INJN A (NUMERAL N0)) (@ε (N -> A -> Prop) (fun z : N -> A -> Prop => True)).
Lemma ZBOT_def {A : Type'} : (@ZBOT A) = (@INJP A (@INJN A (NUMERAL N0)) (@ε (N -> A -> Prop) (fun z : N -> A -> Prop => True))).
Proof. exact (REFL (@ZBOT A)). Qed.
Definition FNIL {A : Type'} : N -> A := fun _17624 : N => @ε A (fun x : A => True).
Lemma FNIL_def {A : Type'} : (@FNIL A) = (fun _17624 : N => @ε A (fun x : A => True)).
Proof. exact (REFL (@FNIL A)). Qed.
Definition OUTL {A B : Type'} : (Datatypes.sum A B) -> A := @ε ((prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> A) (fun OUTL' : (prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> A => forall _17649 : prod N (prod N (prod N N)), forall x : A, (OUTL' _17649 (@inl A B x)) = x) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0))))))))))).
Lemma OUTL_def {A B : Type'} : (@OUTL A B) = (@ε ((prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> A) (fun OUTL' : (prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> A => forall _17649 : prod N (prod N (prod N N)), forall x : A, (OUTL' _17649 (@inl A B x)) = x) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))))))).
Proof. exact (REFL (@OUTL A B)). Qed.
Definition OUTR {A B : Type'} : (Datatypes.sum A B) -> B := @ε ((prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> B) (fun OUTR' : (prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> B => forall _17651 : prod N (prod N (prod N N)), forall y : B, (OUTR' _17651 (@inr A B y)) = y) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0))))))))))).
Lemma OUTR_def {A B : Type'} : (@OUTR A B) = (@ε ((prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> B) (fun OUTR' : (prod N (prod N (prod N N))) -> (Datatypes.sum A B) -> B => forall _17651 : prod N (prod N (prod N N)), forall y : B, (OUTR' _17651 (@inr A B y)) = y) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT0 (BIT1 N0)))))))))))).
Proof. exact (REFL (@OUTR A B)). Qed.
Definition _22857 : Prop -> Prop -> Prop -> Prop -> Prop -> Prop -> Prop -> Prop -> Ascii.ascii := fun a0 : Prop => fun a1 : Prop => fun a2 : Prop => fun a3 : Prop => fun a4 : Prop => fun a5 : Prop => fun a6 : Prop => fun a7 : Prop => _mk_char ((fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7).
Lemma _22857_def : _22857 = (fun a0 : Prop => fun a1 : Prop => fun a2 : Prop => fun a3 : Prop => fun a4 : Prop => fun a5 : Prop => fun a6 : Prop => fun a7 : Prop => _mk_char ((fun a0' : Prop => fun a1' : Prop => fun a2' : Prop => fun a3' : Prop => fun a4' : Prop => fun a5' : Prop => fun a6' : Prop => fun a7' : Prop => @CONSTR (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))) (NUMERAL N0) (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))))) a0' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))) a1' (@pair Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop)))) a2' (@pair Prop (prod Prop (prod Prop (prod Prop Prop))) a3' (@pair Prop (prod Prop (prod Prop Prop)) a4' (@pair Prop (prod Prop Prop) a5' (@pair Prop Prop a6' a7'))))))) (fun n : N => @BOTTOM (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop (prod Prop Prop))))))))) a0 a1 a2 a3 a4 a5 a6 a7)).
Proof. exact (REFL _22857). Qed.
Definition ASCII : Prop -> Prop -> Prop -> Prop -> Prop -> Prop -> Prop -> Prop -> Ascii.ascii := _22857.
Lemma ASCII_def : ASCII = _22857.
Proof. exact (REFL ASCII). Qed.
Definition real_of_num : N -> real := fun m : N => mk_real (fun u : prod hreal hreal => treal_eq (treal_of_num m) u).
Lemma real_of_num_def : real_of_num = (fun m : N => mk_real (fun u : prod hreal hreal => treal_eq (treal_of_num m) u)).
Proof. exact (REFL real_of_num). Qed.
Definition real_neg : real -> real := fun x1 : real => mk_real (fun u : prod hreal hreal => exists x1' : prod hreal hreal, (treal_eq (treal_neg x1') u) /\ (dest_real x1 x1')).
Lemma real_neg_def : real_neg = (fun x1 : real => mk_real (fun u : prod hreal hreal => exists x1' : prod hreal hreal, (treal_eq (treal_neg x1') u) /\ (dest_real x1 x1'))).
Proof. exact (REFL real_neg). Qed.
Definition real_add : real -> real -> real := fun x1 : real => fun y1 : real => mk_real (fun u : prod hreal hreal => exists x1' : prod hreal hreal, exists y1' : prod hreal hreal, (treal_eq (treal_add x1' y1') u) /\ ((dest_real x1 x1') /\ (dest_real y1 y1'))).
Lemma real_add_def : real_add = (fun x1 : real => fun y1 : real => mk_real (fun u : prod hreal hreal => exists x1' : prod hreal hreal, exists y1' : prod hreal hreal, (treal_eq (treal_add x1' y1') u) /\ ((dest_real x1 x1') /\ (dest_real y1 y1')))).
Proof. exact (REFL real_add). Qed.
Definition real_mul : real -> real -> real := fun x1 : real => fun y1 : real => mk_real (fun u : prod hreal hreal => exists x1' : prod hreal hreal, exists y1' : prod hreal hreal, (treal_eq (treal_mul x1' y1') u) /\ ((dest_real x1 x1') /\ (dest_real y1 y1'))).
Lemma real_mul_def : real_mul = (fun x1 : real => fun y1 : real => mk_real (fun u : prod hreal hreal => exists x1' : prod hreal hreal, exists y1' : prod hreal hreal, (treal_eq (treal_mul x1' y1') u) /\ ((dest_real x1 x1') /\ (dest_real y1 y1')))).
Proof. exact (REFL real_mul). Qed.
Definition real_le : real -> real -> Prop := fun x1 : real => fun y1 : real => @ε Prop (fun u : Prop => exists x1' : prod hreal hreal, exists y1' : prod hreal hreal, ((treal_le x1' y1') = u) /\ ((dest_real x1 x1') /\ (dest_real y1 y1'))).
Lemma real_le_def : real_le = (fun x1 : real => fun y1 : real => @ε Prop (fun u : Prop => exists x1' : prod hreal hreal, exists y1' : prod hreal hreal, ((treal_le x1' y1') = u) /\ ((dest_real x1 x1') /\ (dest_real y1 y1')))).
Proof. exact (REFL real_le). Qed.
Definition real_inv : real -> real := fun x : real => mk_real (fun u : prod hreal hreal => exists x' : prod hreal hreal, (treal_eq (treal_inv x') u) /\ (dest_real x x')).
Lemma real_inv_def : real_inv = (fun x : real => mk_real (fun u : prod hreal hreal => exists x' : prod hreal hreal, (treal_eq (treal_inv x') u) /\ (dest_real x x'))).
Proof. exact (REFL real_inv). Qed.
Definition real_sub : real -> real -> real := fun _24026 : real => fun _24027 : real => real_add _24026 (real_neg _24027).
Lemma real_sub_def : real_sub = (fun _24026 : real => fun _24027 : real => real_add _24026 (real_neg _24027)).
Proof. exact (REFL real_sub). Qed.
Definition real_lt : real -> real -> Prop := fun _24038 : real => fun _24039 : real => ~ (real_le _24039 _24038).
Lemma real_lt_def : real_lt = (fun _24038 : real => fun _24039 : real => ~ (real_le _24039 _24038)).
Proof. exact (REFL real_lt). Qed.
Definition real_ge : real -> real -> Prop := fun _24050 : real => fun _24051 : real => real_le _24051 _24050.
Lemma real_ge_def : real_ge = (fun _24050 : real => fun _24051 : real => real_le _24051 _24050).
Proof. exact (REFL real_ge). Qed.
Definition real_gt : real -> real -> Prop := fun _24062 : real => fun _24063 : real => real_lt _24063 _24062.
Lemma real_gt_def : real_gt = (fun _24062 : real => fun _24063 : real => real_lt _24063 _24062).
Proof. exact (REFL real_gt). Qed.
Definition real_abs : real -> real := fun _24074 : real => @COND real (real_le (real_of_num (NUMERAL N0)) _24074) _24074 (real_neg _24074).
Lemma real_abs_def : real_abs = (fun _24074 : real => @COND real (real_le (real_of_num (NUMERAL N0)) _24074) _24074 (real_neg _24074)).
Proof. exact (REFL real_abs). Qed.
Definition real_pow : real -> N -> real := @ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> real -> N -> real) (fun real_pow' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> real -> N -> real => forall _24085 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), (forall x : real, (real_pow' _24085 x (NUMERAL N0)) = (real_of_num (NUMERAL (BIT1 N0)))) /\ (forall x : real, forall n : N, (real_pow' _24085 x (N.succ n)) = (real_mul x (real_pow' _24085 x n)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 (BIT1 N0))))))))))))))).
Lemma real_pow_def : real_pow = (@ε ((prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> real -> N -> real) (fun real_pow' : (prod N (prod N (prod N (prod N (prod N (prod N (prod N N))))))) -> real -> N -> real => forall _24085 : prod N (prod N (prod N (prod N (prod N (prod N (prod N N)))))), (forall x : real, (real_pow' _24085 x (NUMERAL N0)) = (real_of_num (NUMERAL (BIT1 N0)))) /\ (forall x : real, forall n : N, (real_pow' _24085 x (N.succ n)) = (real_mul x (real_pow' _24085 x n)))) (@pair N (prod N (prod N (prod N (prod N (prod N (prod N N)))))) (NUMERAL (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N (prod N N))))) (NUMERAL (BIT1 (BIT0 (BIT1 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N (prod N N)))) (NUMERAL (BIT1 (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N (prod N N))) (NUMERAL (BIT0 (BIT0 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (@pair N (prod N (prod N N)) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 N0)))))))) (@pair N (prod N N) (NUMERAL (BIT0 (BIT0 (BIT0 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))) (@pair N N (NUMERAL (BIT1 (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 N0)))))))) (NUMERAL (BIT1 (BIT1 (BIT1 (BIT0 (BIT1 (BIT1 (BIT1 N0)))))))))))))))).
Proof. exact (REFL real_pow). Qed.
Definition real_div : real -> real -> real := fun _24086 : real => fun _24087 : real => real_mul _24086 (real_inv _24087).
Lemma real_div_def : real_div = (fun _24086 : real => fun _24087 : real => real_mul _24086 (real_inv _24087)).
Proof. exact (REFL real_div). Qed.
Definition real_max : real -> real -> real := fun _24098 : real => fun _24099 : real => @COND real (real_le _24098 _24099) _24099 _24098.
Lemma real_max_def : real_max = (fun _24098 : real => fun _24099 : real => @COND real (real_le _24098 _24099) _24099 _24098).
Proof. exact (REFL real_max). Qed.
Definition real_min : real -> real -> real := fun _24110 : real => fun _24111 : real => @COND real (real_le _24110 _24111) _24110 _24111.
Lemma real_min_def : real_min = (fun _24110 : real => fun _24111 : real => @COND real (real_le _24110 _24111) _24110 _24111).
Proof. exact (REFL real_min). Qed.
Definition real_sgn : real -> real := fun _26598 : real => @COND real (real_lt (real_of_num (NUMERAL N0)) _26598) (real_of_num (NUMERAL (BIT1 N0))) (@COND real (real_lt _26598 (real_of_num (NUMERAL N0))) (real_neg (real_of_num (NUMERAL (BIT1 N0)))) (real_of_num (NUMERAL N0))).
Lemma real_sgn_def : real_sgn = (fun _26598 : real => @COND real (real_lt (real_of_num (NUMERAL N0)) _26598) (real_of_num (NUMERAL (BIT1 N0))) (@COND real (real_lt _26598 (real_of_num (NUMERAL N0))) (real_neg (real_of_num (NUMERAL (BIT1 N0)))) (real_of_num (NUMERAL N0)))).
Proof. exact (REFL real_sgn). Qed.
Definition sqrt : real -> real := fun _27149 : real => @ε real (fun y : real => ((real_sgn y) = (real_sgn _27149)) /\ ((real_pow y (NUMERAL (BIT0 (BIT1 N0)))) = (real_abs _27149))).
Lemma sqrt_def : sqrt = (fun _27149 : real => @ε real (fun y : real => ((real_sgn y) = (real_sgn _27149)) /\ ((real_pow y (NUMERAL (BIT0 (BIT1 N0)))) = (real_abs _27149)))).
Proof. exact (REFL sqrt). Qed.
