Require Import HB.structures.
From mathcomp Require Import ssreflect fintype eqtype ssrfun ssrbool classical.classical_sets boolp choice.
Set Implicit Arguments.
Unset Strict Implicit.
Unset Printing Implicit Defensive.

(* Exact same as isPointed, but we will derive choice and decidable
   equality for it, which could be bad for types that have
   their own choice/eq defined in ssreflect (like nat) if it were derived
   from isPointed directly, because it has an instance of isPointed,
   which would make it so it has two defined equalities.

   This should only be used for types without a predefined decidable equality *)

(* TODO : What of a type like option, which has a pointedType instance
   already defined only if the base type is a choiceType ? we need to define
   option as a Type' in all cases, which means conflicts for (for example) option bool *)

HB.factory Record HOL_isPointed T := {point : T}.

Notation is_Type' := (HOL_isPointed.Build _).

(* in classical context, is a factory for pointedType *)
HB.builders Context T of HOL_isPointed T.

HB.instance Definition _ := isPointed.Build _ point.

HB.instance Definition _ := gen_eqMixin T.

HB.instance Definition _ := gen_choiceMixin T.

HB.end.

Notation Type' := pointedType.
(* Type' is the type of Types of HOL-Light (HOL-Light considers pointed types) *)
(* To define an instance of Type' for a non-empty type [T], use :
   [HB.instance Definition _ := is_Type' a] for some [a : T] *)
